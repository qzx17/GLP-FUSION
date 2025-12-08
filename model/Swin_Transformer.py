import torch
import torch.nn as nn
from timm import create_model
import torch.nn.functional as F

class StudentEngagementSwinV2(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super(StudentEngagementSwinV2, self).__init__()
        
        # 加载预训练的SwinV2模型
        self.backbone = create_model(
            'swinv2_cr_small_ns_224',
            pretrained=pretrained,
            num_classes=0  # 移除原始分类头
        )
        
        # 获取特征维度
        feature_dim = self.backbone.num_features  # 768
        
        # 添加时序处理层
        self.temporal_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        # 添加自定义分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(0.2),
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        if pretrained:
            self._freeze_stages()
    
    def _freeze_stages(self):
        for name, param in self.backbone.named_parameters():
            if "layers.0" in name or "layers.1" in name:
                param.requires_grad = False
    
    def forward(self, x):
        # x shape: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        
        # 重塑输入以处理每个帧
        x = x.view(B*T, C, H, W)
        
        # 提取特征
        features = self.backbone(x)  # [B*T, feature_dim]
        
        # 重塑特征以处理时序
        features = features.view(B, T, -1)  # [B, T, feature_dim]
        
        # 转置以适应时序池化
        features = features.transpose(1, 2)  # [B, feature_dim, T]
        
        # 时序池化
        features = self.temporal_pool(features)  # [B, feature_dim]
        
        # 分类
        out = self.classifier(features)
        
        return out