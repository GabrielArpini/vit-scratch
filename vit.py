import torch
import torch.nn as nn 
from einops import rearrange


class FeedForward(nn.Module):
    def __init__(self,dim, hidden_dim):
        super().__init__()
        self.seq = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.seq(x)

class Attention(nn.Module):
    def __init__(self, d_model, h=8):
        super().__init__()
        assert d_model % heads == 0, "d_model must be divisible by heads"
        self.d_k = d_model // h 
        self.scale = d_k ** -0.5
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.heads = heads
        self.out = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self,x):
        qkv = self.qkv.chunk(3,dim=-1) # this chunks last dimension: (batch, n, d_model*3)
            


class Transformer(nn.Module)
    def __init__(self, dim):
        self.scale = dim ** -0.5


class ViT(nn.Module):
    def __init__(self, P: int):
        super(ViT, self).__init__()
        patch_height, patch_width = (P,P)

        
