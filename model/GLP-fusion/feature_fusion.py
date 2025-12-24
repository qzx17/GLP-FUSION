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

 
class PyramidFusionModule(nn.Module):
    def __init__(self, input_dim=64, depths=[6,4,2], heads=[8,4,2], dim_head=64):
        super().__init__()
        
        self.T_init = 16
        self.fusion_dim = 128
        self.time_scales = [16, 8, 4, 2]
        
        # 多尺度Transformer
        self.transformers = nn.ModuleList([
            Transformer(
                dim=self.fusion_dim,
                depth=depth,
                heads=head,
                dim_head=dim_head,
                mlp_dim=self.fusion_dim*2
            ) for i, (depth, head) in enumerate(zip(depths, heads))
        ])
        
        # ===== 核心修正：down_projections仅保留Conv1d =====
        self.down_projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=self.fusion_dim,
                    out_channels=self.fusion_dim,
                    kernel_size=2,
                    stride=2,
                    padding=0,
                    groups=self.fusion_dim
                )
            ) for _ in range(3)
        ])
        
        # 单独定义归一化/激活/正则化（统一管理）
        self.layer_norm = nn.LayerNorm(self.fusion_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        # 跨尺度融合
        self.cross_scale_fusion = nn.Sequential(
            nn.Linear(self.fusion_dim * 3, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 4)
        )
              
        # 时序位置编码/CLS Token
        self.temporal_pos_embedding = nn.Parameter(torch.randn(1, self.T_init, self.fusion_dim))
        self.temporal_cls_token = nn.Parameter(torch.randn(1, 1, self.fusion_dim))

    def forward(self, x1, x2, return_visuals=False):
        pyramid_cls = []
        pyramid_visuals = {}
        b = x1.shape[0]

        feat = torch.cat([x1, x2], dim=-1)  # [b,16,128]
        feat = feat + self.temporal_pos_embedding

        # ===== 第1层 =====
        feat_perm = feat.permute(0, 2, 1)  # [b,128,16]
        feat_down_8 = self.down_projections[0](feat_perm)  # [b,128,8]
        feat_down_8 = feat_down_8.permute(0, 2, 1)  # [b,8,128]
        feat_down_8 = self.layer_norm(feat_down_8)
        feat_down_8 = self.relu(feat_down_8)
        feat_8 = self.dropout(feat_down_8)
        cls_token = repeat(self.temporal_cls_token, '1 1 d -> b 1 d', b=b)
        feat_8_with_cls = torch.cat([cls_token, feat_8], dim=1)
        feat_8_trans, attn_8 = self.transformers[0](feat_8_with_cls)
        pyramid_cls.append(feat_8_trans[:, 0:1, :])
        pyramid_visuals['low_level_attention'] = attn_8

        # ===== 第2层 =====
        feat_8_perm = feat_8.permute(0, 2, 1)  # [b,128,8]
        feat_down_4 = self.down_projections[1](feat_8_perm)  # [b,128,4]
        feat_down_4 = feat_down_4.permute(0, 2, 1)  # [b,4,128]
        feat_down_4 = self.layer_norm(feat_down_4)
        feat_down_4 = self.relu(feat_down_4)
        feat_4 = self.dropout(feat_down_4)        
        feat_4_with_cls = torch.cat([cls_token, feat_4], dim=1)
        feat_4_trans, attn_4 = self.transformers[1](feat_4_with_cls)
        pyramid_cls.append(feat_4_trans[:, 0:1, :])
        pyramid_visuals['mid_level_attention'] = attn_4

        # ===== 第3层 =====
        feat_4_perm = feat_4.permute(0, 2, 1)  # [b,128,4]
        feat_down_2 = self.down_projections[2](feat_4_perm)  # [b,128,2]
        feat_down_2 = feat_down_2.permute(0, 2, 1)  # [b,2,128]
        feat_down_2 = self.layer_norm(feat_down_2)
        feat_down_2 = self.relu(feat_down_2)
        feat_2 = self.dropout(feat_down_2)        
        feat_2_with_cls = torch.cat([cls_token, feat_2], dim=1)
        feat_2_trans, attn_2 = self.transformers[2](feat_2_with_cls)
        pyramid_cls.append(feat_2_trans[:, 0:1, :])
        pyramid_visuals['high_level_attention'] = attn_2

        # 跨尺度融合
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
            image_size=(224, 224),  # 注意：image_size 应该是一个元组
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
