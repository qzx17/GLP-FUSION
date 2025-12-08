import math
import torch.nn.functional as F

import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange
import numpy as np
from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True


class TS_ConvModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 64, (1, 25), (1, 1)),
            nn.Conv2d(64, 64, (6, 1), (1, 1)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 16), stride=(1, 16)),
            nn.Dropout(0.5),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(64, 64, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class TS_AttentionModule(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** 0.5
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class TS_ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class TS_FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )
        
        
class GELU(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))
    
    
class TS_TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size, num_heads=8, drop_p=0.5, forward_expansion=4, forward_drop_p=0.5):
        super().__init__(
            TS_ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                TS_AttentionModule(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            TS_ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                TS_FeedForwardBlock(emb_size, forward_expansion, forward_drop_p),
                nn.Dropout(drop_p)
            ))
        )


class TS_TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        """
        Initialize the TS_TransformerEncoder.

        Parameters:
            depth (int): Number of sequential Transformer Encoder Blocks.
            emb_size (int): Embedding size used throughout the transformer.
        """        
        super().__init__(*[TS_TransformerEncoderBlock(emb_size) for _ in range(depth)])


class TS_Stream(nn.Module):
    def __init__(self, depth=6, emb_size=64):
        super().__init__()
        self.TS_ConvModule = TS_ConvModule()
        self.TS_TransformerEncoder = TS_TransformerEncoder(depth, emb_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0,2,3,1)
        
        x = self.TS_ConvModule(x)
        x = self.TS_TransformerEncoder(x)
        return x
