import numpy as np
import torch, os
import torch.nn.functional as F
from torch import nn, einsum
from torch.utils.data import Dataset
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from PIL import Image
# import albumentations as A
# from albumentations.pytorch import ToTensorV2


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)



class FeedForward(nn.Module):
    def __init__(self, dim, hid_dim, dropout=0):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(dim, hid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ff(x)



class MHAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5 # "scaled" dot product

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t : rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale # scaled dot product

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)



class Encoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PreNorm(dim, MHAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
                ])
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        
        return x



class TGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(80*80, 8*8*1024)
        self.stage1 = nn.ModuleList([
            Encoder(8*8*1024, 6, 8, 8*8*1024, 8*8*1024),
            Encoder(8*8*1024, 6, 8, 8*8*1024, 8*8*1024),
            Encoder(8*8*1024, 6, 8, 8*8*1024, 8*8*1024),
            Encoder(8*8*1024, 6, 8, 8*8*1024, 8*8*1024),
            Encoder(8*8*1024, 6, 8, 8*8*1024, 8*8*1024)
        ])

        self.ps1 = nn.PixelShuffle(2)

        self.stage2 = nn.ModuleList([
            Encoder(16*16*256, 6, 8, 16*16*256, 16*16*256),
            Encoder(16*16*256, 6, 8, 16*16*256, 16*16*256),
            Encoder(16*16*256, 6, 8, 16*16*256, 16*16*256),
            Encoder(16*16*256, 6, 8, 16*16*256, 16*16*256)
        ])

        self.ps2 = nn.PixelShuffle(4)

        self.stage3 = nn.ModuleList([
            Encoder(16*16*256, 6, 8, 32*32*64, 32*32*64),
            Encoder(16*16*256, 6, 8, 32*32*64, 32*32*64)
        ])

        self.out = nn.Linear(32*32*64, 80*80*3)

    def forward(self, noise):
        x = self.input_layer(noise)
        x = self.stage1(x)
        x = self.ps1(x)
        x = self.stage2(x)
        x = self.ps2(x)
        x = self.stage3(x)
        out = self.out(x)

        return out