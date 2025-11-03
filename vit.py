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
        assert d_model % h == 0, "d_model must be divisible by heads"
        self.d_k = d_model // h
        self.scale = self.d_k ** -0.5
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.h = h
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.softmax = nn.Softmax(dim = -1)

    def forward(self,x):
        qkv = self.qkv(x).chunk(3,dim=-1) # this chunk last dimension: (batch, n, d_model*3)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.h), qkv)    
        dot_prod = torch.matmul(q,k.transpose(-1,-2))
        scaled_prod = dot_prod * self.scale
        sm_scaled_prod = self.softmax(scaled_prod)
        out = torch.matmul(sm_scaled_prod,v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.out(out)

class Transformer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** -0.5


class ViT(nn.Module):
    def __init__(self, P: int):
        super(ViT, self).__init__()
        patch_height, patch_width = (P,P)

        
