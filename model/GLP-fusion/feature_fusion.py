from .temporal_spatial_stream import TS_Stream
import torch
from torch import nn
from torch.backends import cudnn
from .cnn_tvit import TemporalViT,Transformer
from .SLDenseLayer import SLEngagementNet
from .temporal_spatial_stream import TS_Stream
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
        return final_class

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
        self.temporal_pos_embedding = nn.Parameter(torch.randn(1, 16, 128))  # 假设固定16个时间步
        
        # 时序CLS token
        self.temporal_cls_token = nn.Parameter(torch.randn(1, 1, 128))

    def forward(self, x1, x2):
        # x1, x2: [b, 16, 64]
        pyramid_features = []

        # 1. 低分辨率特征处理
        low_level = torch.cat([x1, x2], dim=-1)  # [b, 16, 128]
        low_level = low_level + self.temporal_pos_embedding  # [b, 16, 128]
        b = low_level.shape[0]
        temp_cls_tokens = repeat(self.temporal_cls_token, '1 n d -> b n d', b=b)
        multi_scale_features = torch.cat((temp_cls_tokens, low_level), dim=1)  # [b, 17, 128]
        low_feat = self.transformers[0](low_level)  # [b, 17, 128]
        low_fused = self.fusion_layers[0](low_feat)  # [b, 17, 64]
        pyramid_features.append(low_fused)
        
        # 2. 中等分辨率特征处理
        mid_level = self.down_projections[0](low_feat)  # [b, 17, 32]
        mid_feat = self.transformers[1](mid_level)        
        mid_fused = self.fusion_layers[1](mid_feat)
        pyramid_features.append(mid_fused)
        
        # 3. 高分辨率特征处理
        high_level = self.down_projections[1](mid_feat)  # [b, 17, 16]
        high_feat = self.transformers[2](high_level)
        high_fused = self.fusion_layers[2](high_feat)
        pyramid_features.append(high_fused)
        
        # 4. 跨尺度特征融合
        multi_scale_features = torch.cat(pyramid_features, dim=-1)  # [b, 17, 112]
        
        temporal_features = multi_scale_features[:, 0, :]  # [b, 112]
        
        # 5. 最终分类
        output = self.cross_scale_fusion(temporal_features)  # [b, 4]
        
        return output

class Decision_Fusion(nn.Module):
    def __init__(self, n_classes):
        super(Decision_Fusion, self).__init__()
        self.ct_vit = TemporalViT(
            image_size=(224, 224),  # 注意：image_size 应该是一个元组
            patch_size=16,
            dim=768,
            depth=6,
            heads=12,
            mlp_dim=128,
            # 其他参数如果使用默认值则可以省略
        )
        
        # Model components for each modality
        self.ex_stream = TS_Stream()
        self.slengagement=SLEngagementNet()
         # 使用新的金字塔融合模块
        self.fusion_features = PyramidFusionModule(
            input_dim=128,
            depths=[2,2,2],
            heads=[2,2,2]
        )
        self.classification_head_TS = None
        self.final_fusion = EnhancedFusion()
        
    def forward(self, Img_input, Extract_input, HT_input):
        visual_features = self.ct_vit(Img_input)  # [b, 16, 64]
        behavior_features = self.ex_stream(Extract_input)  # [b, 16, 64]
        sl_features = self.slengagement(HT_input)
        # 金字塔特征融合
        fused_features = self.fusion_features(visual_features, behavior_features)
        final = self.final_fusion(fused_features, sl_features)
        return final
