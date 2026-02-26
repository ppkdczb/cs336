from math import sqrt

from sympy import Si
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from einops import einsum, rearrange, reduce, repeat
import torch.nn.functional as F

class linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.W = torch.nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        std = (2.0 / (in_features + out_features)) ** 0.5
        trunc_normal_(self.W, std=std, a = -3 * std, b = 3 * std)

    def forward(self, x):
        return einsum(self.W, x, 'o i, ... i -> ... o')
    

class embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        trunc_normal_(self.weight, std=1, a = -3, b = 3)

    def forward(self, token_ids: torch.Tensor)-> torch.Tensor:
        return self.weight[token_ids]
    

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.weight = torch.nn.Parameter(torch.empty((d_model,), device=device, dtype=dtype))
    def forward(self, x):
        in_type = x.dtype
        x = x.to(torch.float32)
        sum = einsum(x, x, 'b s d, b s d -> b s') # sum of squares along the last dimension [b, s]
        rms = rearrange(torch.sqrt(sum / self.d_model + self.eps), 'b s -> b s 1')
        result = x * self.weight / rms
        return result.to(in_type)
        
class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class SwiGLU_FFN(nn.Module):
    def __init__(self, d_model: int, device=None, dtype=None):
        super().__init__()
        d_ff = d_model * 3 // 8
        d_ff = (d_ff + 63) // 64 * 64
        self.W1 = linear(d_model, d_ff, device=device, dtype=dtype)
        self.W2 = linear(d_ff, d_model, device=device, dtype=dtype)
        self.W3 = linear(d_model, d_ff, device=device, dtype=dtype)
        self.SiLU = SiLU()

    def forward(self, x):
        return self.W2(self.SiLU(self.W1(x)) * self.W3(x))

class RoPE(nn.Module):
    def __init__(self, theta:float, d_K: int,max_seq_len: int, device=None, dtype=None):
            super().__init__()
            self.d_K = d_K
            self.max_seq_len = max_seq_len
            
            # 1. 生成位置索引 i [0, 1, 2, ..., max_seq_len-1]
            # shape: (max_seq_len,)
            position = torch.arange(max_seq_len, device=device, dtype=torch.float32)
            
            # 2. 生成维度索引 k [0, 1, ..., d_K//2 - 1]
            # 并计算分母的频率底数: theta ** ((2*k) / d_K)
            # shape: (d_K // 2,)
            k = torch.arange(d_K // 2, device=device, dtype=torch.float32)
            inv_freq = 1.0 / (theta ** ((2 * k) / d_K))
            
            # 3. 计算所有的 theta 角度 (利用外积，自动相乘)
            # i 的形状 (max_seq_len, 1) 乘以 inv_freq 的形状 (d_K // 2,)
            # 结果 freqs 的形状将直接是 (max_seq_len, d_K // 2)
            freqs = torch.einsum('i,j->ij', position, inv_freq)
            
            # 4. 直接求 cos 和 sin
            cos_cached = torch.cos(freqs).to(dtype)
            sin_cached = torch.sin(freqs).to(dtype)
            
            self.register_buffer('cos_cached', cos_cached, persistent=False)
            self.register_buffer('sin_cached', sin_cached, persistent=False)

    def forward(self, x):
        # x:[..., seq_len, d_K]
        x_even = x[..., 0::2]
        x_odd  = x[..., 1::2]
        x_rev = torch.stack([-x_odd, x_even], dim=-1).flatten(-2)
        seq_len = x.shape[-2]
        emb_cos = self.cos_cached[:seq_len]
        emb_sin = self.sin_cached[:seq_len]
        repeat_cos = torch.repeat_interleave(emb_cos, repeats=2, dim=-1)
        repeat_sin = torch.repeat_interleave(emb_sin, repeats=2, dim=-1)
        return x * repeat_cos + x_rev * repeat_sin