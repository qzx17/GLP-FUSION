import torch
import torch.nn as nn
import torch.nn.functional as F

class SLBlock(nn.Module):
    def __init__(self, input_dim, output_dim, expansion):
        super().__init__()
        hidden_dim = input_dim * expansion
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class SLEngagementNet(nn.Module):
    def __init__(self, input_dim=4, output_dim=4, expansion=4):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.sl_block = SLBlock(input_dim, output_dim, expansion)
        
        # 添加残差连接的投影层（因为输入输出维度不同）
        self.residual_proj = nn.Linear(input_dim, output_dim)
        
        # 最终的归一化层
        self.final_norm = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        # 输入x的形状: (batch_size, input_dim) = (32, 4)
        b=x.shape[0]
        x = self.norm(x)
        
        # 主路径
        main = self.sl_block(x)
        
        # 残差路径（需要投影因为维度不同）
        residual = self.residual_proj(x)
        
        # 残差连接
        out = main + residual
        
        # 最终归一化
        out = self.final_norm(out)
        
        out = out.view(b,-1)
        # 使用softmax使输出离散化
        out = F.softmax(out, dim=-1)
        
        return out  # 输出形状: (batch_size, output_dim) = (32, 64)