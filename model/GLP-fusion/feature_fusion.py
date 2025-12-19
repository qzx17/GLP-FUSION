from .temporal_spatial_stream import TS_Stream
import torch
from torch import nn
from torch.backends import cudnn
from .cnn_tvit import TemporalViT,Transformer
from .SLDenseLayer import SLEngagementNet
from .ts_eye_pose import TS_Stream
cudnn.benchmark = False
cudnn.deterministic = True
import torch.nn.functional as F
from einops import rearrange, repeat
from torchvision import models
class EnhancedFusion(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, input_dim),
            nn.Sigmoid()
        )       
    
        self.linear = nn.Linear(input_dim, 4)
    def forward(self, x1, x2):
        combined = torch.cat([x1, x2], dim=-1)
        weights = self.attention(combined)  # 学习权重
        weighted_features = combined * weights  # 加权融合
        final_class = self.linear(weighted_features)
        return final_class, weights

class AdaptationGate(nn.Module):
    def __init__(self, feature_dim=4, hidden_dim=64, num_classes=4, dropout_rate=0.2):
        super().__init__()
        
        # 特征融合部分
        self.gate_net = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),  # 连接两个特征
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 特征变换
        self.transform_v = nn.Linear(feature_dim, hidden_dim)
        self.transform_s = nn.Linear(feature_dim, hidden_dim)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, V, S):
        """
        Args:
            V: 视觉特征 (batch_size, feature_dim)
            S: 生理特征 (batch_size, feature_dim)
        Returns:
            output: 分类结果 (batch_size, num_classes)
        """
        # 确保输入维度正确
        assert V.dim() == 2 and S.dim() == 2, f"输入维度必须是2D，当前维度: V-{V.dim()}, S-{S.dim()}"
        assert V.size(0) == S.size(0), f"批次大小不匹配: V-{V.size(0)}, S-{S.size(0)}"
        
        # 计算注意力权重
        gate = self.gate_net(torch.cat([V, S], dim=1))
        
        # 特征变换
        V_trans = self.transform_v(V)
        S_trans = self.transform_s(S)
        
        # 特征融合
        fused = gate * S_trans + V_trans
        
        # 分类
        output = self.classifier(fused)
        
        return output

 
class PyramidFusionModule(nn.Module):
    def __init__(self, input_dim=64, depths=[6,4,2], heads=[8,4,2], dim_head=64):
        super().__init__()
        
        # 多尺度特征维度
        self.dims = [64, 32, 16]  # 从低维维到高维
        
        # 多尺度 Transformer
        self.transformers = nn.ModuleList([
            Transformer(
                dim=128 if i==0 else self.dims[i-1],  # 第一层处理128维输入
                depth=depth,
                heads=head,
                dim_head=dim_head,
                mlp_dim=dim*2
            ) for i, (dim, depth, head) in enumerate(zip(self.dims, depths, heads))
        ])
        
        # 特征转换层
        self.down_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128 if i==0 else self.dims[i-1], self.dims[i]),
                nn.LayerNorm(self.dims[i]),
                nn.ReLU(),
                nn.Dropout(0.3)
            ) for i in range(len(self.dims)-1)
        ])
        
        # 特征融合层
        self.fusion_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128 if i==0 else self.dims[i-1], self.dims[i]),
                nn.LayerNorm(self.dims[i]),
                nn.ReLU(),
                nn.Dropout(0.3)
            ) for i in range(len(self.dims))
        ])
        
        # 跨尺度融合
        fusion_input_dim = sum(self.dims)  # 64 + 32 + 16 = 112
        self.cross_scale_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64,4),
            nn.LayerNorm(4)
        )
              
        # 时序位置编码
        self.temporal_pos_embedding = nn.Parameter(torch.randn(1, 16, 128))  # 固定16个时间步
        
        # 时序CLS token
        self.temporal_cls_token = nn.Parameter(torch.randn(1, 1, 128))

    def forward(self, x1, x2, return_visuals=False):
        # x1, x2: [b, 16, 64]
        pyramid_features = []
        pyramid_visuals = {}

        # 1. 低分辨率特征处理
        low_level = torch.cat([x1, x2], dim=-1)  # [b, 16, 128]
        low_level = low_level + self.temporal_pos_embedding  # [b, 16, 128]
        b = low_level.shape[0]
        temp_cls_tokens = repeat(self.temporal_cls_token, '1 n d -> b n d', b=b)
        multi_scale_features = torch.cat((temp_cls_tokens, low_level), dim=1)  # [b, 17, 128]
        low_feat, low_attn = self.transformers[0](low_level)  # [b, 17, 128]
        low_fused = self.fusion_layers[0](low_feat)  # [b, 17, 64]
        pyramid_features.append(low_fused)
        pyramid_visuals['low_level_attention'] = low_attn # 存入字典

        # 2. 中等分辨率特征处理
        mid_level = self.down_projections[0](low_feat)  # [b, 17, 32]
        mid_feat, mid_attn = self.transformers[1](mid_level)        
        mid_fused = self.fusion_layers[1](mid_feat)
        pyramid_features.append(mid_fused)
        pyramid_visuals['mid_level_attention'] = mid_attn
        
        # 3. 高分辨率特征处理
        high_level = self.down_projections[1](mid_feat)  # [b, 17, 16]
        high_feat, high_attn = self.transformers[2](high_level)
        high_fused = self.fusion_layers[2](high_feat)
        pyramid_features.append(high_fused)
        pyramid_visuals['high_level_attention'] = high_attn
        
        # 4. 跨尺度特征融合
        multi_scale_features = torch.cat(pyramid_features, dim=-1)  # [b, 17, 112]
        
        temporal_features = multi_scale_features[:, 0, :]  # [b, 112]

        output = self.cross_scale_fusion(temporal_features)  # [b, 4]
        
        if return_visuals:
            return output, pyramid_visuals
        else:
            return output

class Decision_Fusion(nn.Module):
    def __init__(self, n_classes):
        super(Decision_Fusion, self).__init__()
        self.ct_vit = TemporalViT(
            image_size=(112, 112),  # 注意：image_size 应该是一个元组
            patch_size=16,
            dim=768,
            depth=6,
            heads=12,
            mlp_dim=128,
        )
        self.ex_stream = TS_Stream()
        self.slengagement=SLEngagementNet()
        self.fusion_features = PyramidFusionModule(
            input_dim=128,
            depths=[2,2,2],
            heads=[2,2,2]
        )
        self.final_fusion = EnhancedFusion()


    def forward(self, Img_input, Extract_input, HT_input, return_visuals=False):

        # 1. 调用 TemporalViT，并根据 return_visuals 接收输出
        if return_visuals:
            visual_features, vit_visuals = self.ct_vit(Img_input, return_visuals=True)
        else:
            visual_features = self.ct_vit(Img_input, return_visuals=False)

        behavior_features = self.ex_stream(Extract_input)
        sl_features = self.slengagement(HT_input)
        # return sl_features
        
        if return_visuals:
            fused_features, pyramid_visuals = self.fusion_features(visual_features, behavior_features, return_visuals)
        else:
            fused_features = self.fusion_features(visual_features, behavior_features, return_visuals)
        
        final_output, fusion_weights = self.final_fusion(fused_features, sl_features)

        # ----------------- 整合可视化数据 -----------------
        if return_visuals:
            all_visuals = {
                # 来自 cnn_vit 的最终 attention
                'cnn_vit_attention': vit_visuals['vit_attention'],                
                # 来自金字塔融合的三个 attention
                'pyramid_low_attention': pyramid_visuals['low_level_attention'],
                'pyramid_mid_attention': pyramid_visuals['mid_level_attention'],
                'pyramid_high_attention': pyramid_visuals['high_level_attention'],

                # 三个原始特征
                'behavior_features': behavior_features,
                'sl_features': sl_features,
                
                # 最终融合权重
                'fusion_weights': fusion_weights,
            }
            return final_output, all_visuals, fused_features
        else:
            return final_output, fused_features
