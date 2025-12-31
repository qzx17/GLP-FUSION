import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU()

        # 残差连接
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out += residual
        out = self.relu3(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.2):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), attn

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),  # Pre-norm 结构
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                nn.LayerNorm(dim),  # Pre-norm 结构
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))
        self.final_norm = nn.LayerNorm(dim)  # 最终的层归一化

    def forward(self, x):
        attn_all = []
        for norm1, attn, norm2, ff in self.layers:
            attn_output, attn_map = attn(norm1(x))
            x = x + attn_output  # Pre-norm + 残差连接
            attn_all.append(attn_map)
            x = x + ff(norm2(x))
        return self.final_norm(x), attn_all[-1]
    
class TemporalViT(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, pool='cls', channels=64, dim_head=64, dropout=0.3, emb_dropout=0.2):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # 残差块
        self.residual_block = ResidualBlock(32, 64, kernel_size=3, stride=1, padding=1)
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.temporal_projection = nn.Sequential(
            nn.Linear(dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, x, return_visuals=False):
        b, t, n, h, w = x.shape
        x_reshaped = x.reshape(b * t, n, h, w)

        # 1. 提取CNN特征图
        cnn_features = self.init_conv(x_reshaped)
        cnn_features = self.residual_block(cnn_features) # 形状 [b*t, 64, H, W]

        _, n, h, w = cnn_features.shape
        x_patched = cnn_features.reshape(b, -1, n, h, w) # 重新组合 batch 和 time 维度
        x_embedded = self.to_patch_embedding(x_patched)
        
        b, t, n_patches, _ = x_embedded.shape
        
        x_flat = x_embedded.reshape(b * t, n_patches, -1)

        cls_tokens = repeat(self.cls_token, '1 1 1 d -> bt 1 d', bt = b * t)
        x_with_cls = torch.cat((cls_tokens, x_flat), dim=1)
        x_with_cls += self.pos_embedding[:, :(n_patches + 1), :]
        x_with_cls = self.dropout(x_with_cls)

        # 2. 接收 Transformer 的输出和注意力图
        x_transformed, attn_map = self.transformer(x_with_cls)

        x_reshaped_back = x_transformed.reshape(b, t, (n_patches + 1), -1)
        cls_output = x_reshaped_back[:, :, 0, :]
        final_features = self.temporal_projection(cls_output) # 形状 [b, t, 64]
        
        if return_visuals:
            visuals = {
                'vit_attention': attn_map,      # ViT 注意力图
                'cnn_features': cnn_features,   # CNN 特征图 (for Grad-CAM)
                'final_features': final_features, # 模型最终输出的视觉特征
            }
            # 注意：这里的 final_features 是修改前 Decision_Fusion 中 visual_features 的来源
            return final_features, visuals
        else:
            return final_features
