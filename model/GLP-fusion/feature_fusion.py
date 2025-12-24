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
        
        # ===== 核心参数 =====
        self.T_init = 16  # 初始时间步
        self.fusion_dim = 128  # 拼接后特征维度（64+64）
        self.time_scales = [16, 8, 4, 2]  # 时间降采样目标（16→8→4→2）
        
        # ===== 多尺度 Transformer（D=128不变）=====
        self.transformers = nn.ModuleList([
            Transformer(
                dim=self.fusion_dim,  # 特征维度始终128
                depth=depth,
                heads=head,
                dim_head=dim_head,
                mlp_dim=self.fusion_dim*2
            ) for i, (depth, head) in enumerate(zip(depths, heads))
        ])
        
        # ===== 时间降采样层（仅降T，D=128不变）=====
        # 改用BatchNorm1d（适配Conv1d的[B,D,T]维度，对D维度归一化）
        self.down_projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=self.fusion_dim,
                    out_channels=self.fusion_dim,
                    kernel_size=2,
                    stride=2,
                    padding=0,
                    groups=self.fusion_dim  # 分组卷积：仅降时间维度T，D不变
                ),
                nn.BatchNorm1d(self.fusion_dim),  # ✅ 适配[B,D,T]，对D维度归一化
                nn.ReLU(),
                nn.Dropout(0.3)
            ) for _ in range(3)  # 3层降采样：16→8、8→4、4→2
        ])
        
        # ===== 跨尺度融合（3个CLS拼接：128*3=384 → 4）=====
        self.cross_scale_fusion = nn.Sequential(
            nn.Linear(self.fusion_dim * 3, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 4)
        )
              
        # ===== 时序位置编码/CLS Token（维度[1,T,128]/[1,1,128]）=====
        self.temporal_pos_embedding = nn.Parameter(torch.randn(1, self.T_init, self.fusion_dim))  # [1,16,128]
        self.temporal_cls_token = nn.Parameter(torch.randn(1, 1, self.fusion_dim))  # [1,1,128]

    def forward(self, x1, x2, return_visuals=False):
        pyramid_cls = []
        pyramid_visuals = {}
        b = x1.shape[0]
        
        # ===== 初始特征：拼接+位置编码（T=16，D=128）=====
        feat = torch.cat([x1, x2], dim=-1)  # [b, 16, 128]
        feat = feat + self.temporal_pos_embedding  # 加位置编码，维度匹配

        # ===== 第1步：时间降采样16→8 → 再Transformer =====
        feat_perm = feat.permute(0, 2, 1)  # [b,128,16]
        feat_down_8 = self.down_projections[0](feat_perm)  # [b,128,8]
        feat_8 = feat_down_8.permute(0, 2, 1)  # [b,8,128]
        cls_token = repeat(self.temporal_cls_token, '1 1 d -> b 1 d', b=b)
        feat_8_with_cls = torch.cat([cls_token, feat_8], dim=1)  # [b,9,128]
        feat_8_trans, attn_8 = self.transformers[0](feat_8_with_cls)  # [b,9,128]
        cls_8 = feat_8_trans[:, 0:1, :]  # [b,1,128]
        pyramid_cls.append(cls_8)
        pyramid_visuals['scale_16to8_attention'] = attn_8

        # ===== 第2步：时间降采样8→4 =====
        feat_8_perm = feat_8.permute(0, 2, 1)  # [b,128,8]
        feat_down_4 = self.down_projections[1](feat_8_perm)  # [b,128,4]
        feat_4 = feat_down_4.permute(0, 2, 1)  # [b,4,128]
        feat_4_with_cls = torch.cat([cls_token, feat_4], dim=1)  # [b,5,128]
        feat_4_trans, attn_4 = self.transformers[1](feat_4_with_cls)  # [b,5,128]
        cls_4 = feat_4_trans[:, 0:1, :]  # [b,1,128]
        pyramid_cls.append(cls_4)
        pyramid_visuals['scale_8to4_attention'] = attn_4

        # ===== 第3步：时间降采样4→2 =====
        feat_4_perm = feat_4.permute(0, 2, 1)  # [b,128,4]
        feat_down_2 = self.down_projections[2](feat_4_perm)  # [b,128,2]
        feat_2 = feat_down_2.permute(0, 2, 1)  # [b,2,128]
        feat_2_with_cls = torch.cat([cls_token, feat_2], dim=1)  # [b,3,128]
        feat_2_trans, attn_2 = self.transformers[2](feat_2_with_cls)  # [b,3,128]
        cls_2 = feat_2_trans[:, 0:1, :]  # [b,1,128]
        pyramid_cls.append(cls_2)
        pyramid_visuals['scale_4to2_attention'] = attn_2

        # ===== 拼接3个CLS =====
        fused_cls = torch.cat(pyramid_cls, dim=-1)  # [b,1,384]
        temporal_features = fused_cls.squeeze(1)  # [b,384]
        output = self.cross_scale_fusion(temporal_features)  # [b,4]

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
