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
        self.residual_proj = nn.Linear(input_dim, output_dim)
        self.final_norm = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        b=x.shape[0]
        x = self.norm(x)
        main = self.sl_block(x)
        residual = self.residual_proj(x)
        out = main + residual
        out = self.final_norm(out)        
        out = out.view(b,-1)
        out = F.softmax(out, dim=-1)
        
        return out 
