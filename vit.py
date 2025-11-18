import torch 
import torch.nn as nn 
from einops import rearrange


# Lucidrains positional embedding implementation.

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


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
    def __init__(self, d_model,N, h, ff_hidden_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.layers = nn.ModuleList([])
        for _ in range(N):
            self.layers.append(nn.ModuleList([
                Attention(d_model,h),
                FeedForward(d_model, ff_hidden_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x 
            x = self.norm1(x)
            x = ff(x) + x 
            x = self.norm2(x)
        return x




class ViT(nn.Module):
    def __init__(self, P: int, C: int, d_model:int, H: int, W: int, num_classes: int):
        super(ViT, self).__init__() 
        # Very cool way to extract the patches of the image and also do the Linear projection 
        self.extract_patches = nn.Conv2d(in_channels=C, out_channels=d_model, kernel_size=P, stride=P)
        self.pos_embedding = posemb_sincos_2d(
            h = H // P,
            w = W // P,
            dim = d_model,
        ) 
        self.transformer = Transformer(d_model=d_model,N=8,h=8,ff_hidden_dim=128)
        self.linear_classify = nn.Linear(d_model, num_classes)


    
    def forward(self, x):
        device = x.device 
        x = self.extract_patches(x)
        # extraction will return dim: (b, d_model, p, p).
        x = rearrange(x,'b d h w -> b (h w) d') # Rearrange the extracted dim to: (b, N, d_model)
        x += self.pos_embedding.to(device, dtype=x.dtype) # Add positional embeddings values 
        x = self.transformer(x) # Pass through the transformer architecture 
        x = x.mean(dim=1) # Get the mean of the N values to use in the classification 
        return self.linear_classify(x)
        
    
        
