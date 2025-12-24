from .temporal_spatial_stream import TS_Stream
import torch
from torch import nn
from torch.backends import cudnn
from .cnn_tvit import TemporalViT, Transformer
from .SLDenseLayer import SLEngagementNet
from .ts_eye_pose import TS_Stream
cudnn.benchmark = False
cudnn.deterministic = True
import torch.nn.functional as F
from einops import rearrange, repeat
from torchvision import models

# -------------------------- 1. EnhancedFusion --------------------------
class EnhancedFusion(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=None, output_dim=4, dropout_rate=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else input_dim * 2
        self.output_dim = output_dim
        
        self.attention = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, input_dim),
            nn.Sigmoid()
        )       
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x1, x2):
        combined = torch.cat([x1, x2], dim=-1)
        weights = self.attention(combined)  
        weighted_features = combined * weights  
        weighted_features = self.dropout(weighted_features) 
        final_class = self.linear(weighted_features)
        return final_class, weights

# -------------------------- 2. PyramidFusionModule --------------------------
class PyramidFusionModule(nn.Module):
    def __init__(
        self, 
        input_dim=64, 
        depths=[6,4,2], 
        heads=[8,4,2], 
        dim_head=64,
        T_init=16,        
        dropout_1=0.3,      
        dropout_2=0.2,    
        fusion_linear_dim=64, 
        output_dim=4   
    ):
        super().__init__()
        
        self.T_init = T_init
        self.fusion_dim = input_dim
        self.time_scales = [16, 8, 4, 2]
        
        self.transformers = nn.ModuleList([
            Transformer(
                dim=self.fusion_dim,
                depth=depth,
                heads=head,
                dim_head=dim_head,
                mlp_dim=self.fusion_dim*2
            ) for i, (depth, head) in enumerate(zip(depths, heads))
        ])
        
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
        
        self.layer_norm = nn.LayerNorm(self.fusion_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_1)  
        
        self.cross_scale_fusion = nn.Sequential(
            nn.Linear(self.fusion_dim * 3, fusion_linear_dim),
            nn.LayerNorm(fusion_linear_dim),
            nn.ReLU(),
            nn.Dropout(dropout_2),  
            nn.Linear(fusion_linear_dim, output_dim)
        )
              
        # position embedding/CLS Token
        self.temporal_pos_embedding = nn.Parameter(torch.randn(1, self.T_init, self.fusion_dim))
        self.temporal_cls_token = nn.Parameter(torch.randn(1, 1, self.fusion_dim))

    def forward(self, x1, x2, return_visuals=False):
        pyramid_cls = []
        pyramid_visuals = {}
        b = x1.shape[0]

        feat = torch.cat([x1, x2], dim=-1)  # [b,16,128]
        feat = feat + self.temporal_pos_embedding

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

        fused_cls = torch.cat(pyramid_cls, dim=-1)  # [b,1,384]
        temporal_features = fused_cls.squeeze(1)  # [b,384]
        output = self.cross_scale_fusion(temporal_features)  # [b,4]

        if return_visuals:
            return output, pyramid_visuals
        else:
            return output

# -------------------------- 3. Decision_Fusion --------------------------
class Decision_Fusion(nn.Module):
    def __init__(
        self, 
        n_classes=4,
        # TemporalViT
        vit_image_size=(224,224),
        vit_patch_size=16,
        vit_dim=768,
        vit_depth=6,
        vit_heads=12,
        vit_mlp_dim=128,
        # PyramidFusionModule
        pyramid_input_dim=128,
        pyramid_depths=[2,2,2],
        pyramid_heads=[2,2,2],
        pyramid_dim_head=64,
        pyramid_T_init=16,
        pyramid_dropout_1=0.3,
        pyramid_dropout_2=0.2,
        # EnhancedFusion
        enhanced_input_dim=8,
        enhanced_hidden_dim=None,
        enhanced_dropout=0.0
    ):
        super(Decision_Fusion, self).__init__()
        self.n_classes = n_classes
        
        self.ct_vit = TemporalViT(
            image_size=vit_image_size,
            patch_size=vit_patch_size,
            dim=vit_dim,
            depth=vit_depth,
            heads=vit_heads,
            mlp_dim=vit_mlp_dim,
        )
        self.ex_stream = TS_Stream()
        self.slengagement = SLEngagementNet()
        
        # PyramidFusionModule
        self.fusion_features = PyramidFusionModule(
            input_dim=pyramid_input_dim,
            depths=pyramid_depths,
            heads=pyramid_heads,
            dim_head=pyramid_dim_head,
            T_init=pyramid_T_init,
            dropout_1=pyramid_dropout_1,
            dropout_2=pyramid_dropout_2,
            output_dim=n_classes
        )
        
        # EnhancedFusion
        self.final_fusion = EnhancedFusion(
            input_dim=enhanced_input_dim,
            hidden_dim=enhanced_hidden_dim,
            output_dim=n_classes,
            dropout_rate=enhanced_dropout
        )

    def forward(self, Img_input, Extract_input, HT_input, return_visuals=False):
        if return_visuals:
            visual_features, vit_visuals = self.ct_vit(Img_input, return_visuals=True)
        else:
            visual_features = self.ct_vit(Img_input, return_visuals=False)

        behavior_features = self.ex_stream(Extract_input)
        sl_features = self.slengagement(HT_input)
        
        if return_visuals:
            fused_features, pyramid_visuals = self.fusion_features(visual_features, behavior_features, return_visuals)
        else:
            fused_features = self.fusion_features(visual_features, behavior_features, return_visuals)

        final_output, fusion_weights = self.final_fusion(fused_features, sl_features)

        # 整合可视化数据
        if return_visuals:
            all_visuals = {
                'cnn_vit_attention': vit_visuals['vit_attention'],                
                'pyramid_low_attention': pyramid_visuals.get('low_level_attention'),
                'pyramid_mid_attention': pyramid_visuals.get('mid_level_attention'),
                'pyramid_high_attention': pyramid_visuals.get('high_level_attention'),
                'behavior_features': behavior_features,
                'sl_features': sl_features,
                'fusion_weights': fusion_weights,
            }
            return final_output, all_visuals, fused_features
        else:
            return final_output, fused_features
