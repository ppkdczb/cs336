from calendar import c
from itertools import accumulate
from math import cos, sqrt
from typing import Tuple, Optional, IO, Any, BinaryIO

from sympy import content
from torch import Tensor, ge
from jaxtyping import Bool, Float, Int
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from einops import einsum, rearrange, reduce, repeat
import torch.nn.functional as F
from collections.abc import Callable, Iterable
from numpy import sort, typing as npt
import os


class linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.W = torch.nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        std = (2.0 / (in_features + out_features)) ** 0.5
        trunc_normal_(self.W, std=std, a=-3 * std, b=3 * std)

    def forward(self, x):
        return einsum(self.W, x, 'o i, ... i -> ... o')  # [..., o]


class embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )
        trunc_normal_(self.weight, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.weight = torch.nn.Parameter(
            torch.ones((d_model,), device=device, dtype=dtype)
        )

    def forward(self, x):
        in_type = x.dtype
        x = x.to(torch.float32)
        sum = einsum(
            x, x, 'b s d, b s d -> b s'
        )  # sum of squares along the last dimension [b, s]
        rms = rearrange(torch.sqrt(sum / self.d_model + self.eps), 'b s -> b s 1')
        result = x * self.weight / rms
        return result.to(in_type)


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class SwiGLU_FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int = None, device=None, dtype=None):
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 8 // 3
            d_ff = (d_ff + 63) // 64 * 64
        self.W1 = linear(d_model, d_ff, device=device, dtype=dtype)
        self.W2 = linear(d_ff, d_model, device=device, dtype=dtype)
        self.W3 = linear(d_model, d_ff, device=device, dtype=dtype)
        self.SiLU = SiLU()

    def forward(self, x):
        return self.W2(self.SiLU(self.W1(x)) * self.W3(x))


class RoPE(nn.Module):
    def __init__(
        self, theta: float, d_K: int, max_seq_len: int, device=None, dtype=None
    ):
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
        x_odd = x[..., 1::2]
        x_rev = torch.stack([-x_odd, x_even], dim=-1).flatten(-2)
        seq_len = x.shape[-2]
        emb_cos = self.cos_cached[:seq_len]
        emb_sin = self.sin_cached[:seq_len]
        repeat_cos = torch.repeat_interleave(emb_cos, repeats=2, dim=-1)
        repeat_sin = torch.repeat_interleave(emb_sin, repeats=2, dim=-1)
        return x * repeat_cos + x_rev * repeat_sin


def softmax(x, dim=-1):
    x = x - torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(x)
    sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp_x


def scaled_dot_product_attention(
    Q,
    K,
    V,
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... d_out"]:
    '''
    Q: [b, ..., s, d_k]
    K: [b, ..., s, d_k]
    V: [b, ..., s, d_v]
    '''
    d_k = Q.shape[-1]
    sqrt_d_k = sqrt(d_k)
    scores = einsum(Q, K, 'b ... s1 d_k, b ... s2 d_k -> b ... s1 s2') / sqrt_d_k
    if mask is not None:
        scores = scores.masked_fill(mask == False, float('-inf'))
    attn_weights = softmax(scores, dim=-1)
    output = einsum(attn_weights, V, 'b ... s1 s2 , b ... s2 d_v -> b ... s1 d_v')
    return output


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: Optional[int] = 2048,
        theta: Optional[float] = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        if theta is not None:
            self.rope = RoPE(theta, self.d_k, max_seq_len, device=device, dtype=dtype)
        self.W_Q = linear(d_model, d_model, device=device, dtype=dtype)
        self.W_K = linear(d_model, d_model, device=device, dtype=dtype)
        self.W_V = linear(d_model, d_model, device=device, dtype=dtype)
        self.W_O = linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x):
        b, s, _ = x.shape
        Q = rearrange(self.W_Q(x), '... s (h d_k) -> ... h s d_k', h=self.num_heads)
        K = rearrange(self.W_K(x), '... s (h d_k) -> ... h s d_k', h=self.num_heads)
        V = rearrange(self.W_V(x), '... s (h d_v) -> ... h s d_v', h=self.num_heads)
        if hasattr(self, 'rope'):
            Q = self.rope(Q)
            K = self.rope(K)
        mask = torch.tril(torch.ones((s, s), device=x.device)).bool()
        attn_output = scaled_dot_product_attention(Q, K, V, mask=mask)
        attn_output = rearrange(attn_output, '... h s d_v -> ... s (h d_v)')
        output = self.W_O(attn_output)
        return output


class transformer_block(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model=d_model)
        self.norm2 = RMSNorm(d_model=d_model)
        self.attention = MultiHeadAttention(d_model, num_heads, max_seq_len, theta)
        self.ffn = SwiGLU_FFN(d_model, d_ff)

    def forward(self, x):
        # x: batch sequence_length d_model
        x_norm1 = self.norm1(x)
        attn_output = self.attention(x_norm1)
        x = x + attn_output
        x_norm2 = self.norm2(x)
        ffn_output = self.ffn(x_norm2)
        x = x + ffn_output
        return x


class transformer_lm(nn.Module):

    def __init__(
        self,
        vocab_size: int,  # vocab size
        context_length: int,  # max sequence length
        d_model: int,  # embedding dimension
        num_layers: int,  # number of transformer blocks
        num_heads: int,  # number of attention heads
        d_ff: int,  # feedforward dimension
        rope_theta: float,  # RoPE 的 theta 参数
    ):
        super().__init__()
        self.token_embedding = embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                transformer_block(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(d_model)
        self.fc = linear(d_model, vocab_size)

    def forward(self, token_ids):
        x = self.token_embedding(token_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.fc(x)
        return logits


def log_softmax(x, dim=-1):
    x_max = torch.max(x, dim=dim, keepdim=True).values
    shifted = x - x_max
    return shifted - torch.log(torch.sum(torch.exp(shifted), dim=dim, keepdim=True))


def cross_entropy_loss(logits, target_ids):
    '''
    logits: [...,v], eg:[b,v]
    target_ids: [...,] eg:[b,]
    '''

    flatten_logits = rearrange(logits, '... v -> (...) v')
    flatten_target_ids = rearrange(target_ids, '... -> (...)')
    log_probs = log_softmax(flatten_logits, dim=-1)
    row_idx = torch.arange(flatten_logits.size(0), device=flatten_logits.device)
    loss = -log_probs[row_idx, flatten_target_ids].mean()

    return loss


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self.state[param]
                grad = param.grad.data
                t = state.get("step", 0)
                param.data = param.data - lr / sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(
        self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad.data
                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(param.data)
                    state["v"] = torch.zeros_like(param.data)
                state["step"] += 1
                t = state["step"]
                m = state["m"]
                v = state["v"]
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                alpha = lr * sqrt(1 - beta2**t) / (1 - beta1**t)
                param.data -= alpha * m / (torch.sqrt(v) + eps)
                param.data -= lr * weight_decay * param.data
        return loss


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    elif it >= warmup_iters and it <= cosine_cycle_iters:
        lr = (
            min_learning_rate
            + (max_learning_rate - min_learning_rate)
            * (
                1
                + cos(
                    (it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * torch.pi
                )
            )
            / 2
        )
        return lr
    else:
        return min_learning_rate


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    torch
    clip_coef = max_l2_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(0, len(dataset) - context_length, (batch_size,))
    x_batch = []
    y_batch = []

    for i in starts:
        x_fragment = dataset[i : i + context_length]
        y_fragment = dataset[i + 1 : i + context_length + 1]
        x_batch.append(torch.from_numpy(x_fragment).long())
        y_batch.append(torch.from_numpy(y_fragment).long())

    x_batch = torch.stack(x_batch).to(device)
    y_batch = torch.stack(y_batch).to(device)

    return x_batch, y_batch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:

    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    iteration = checkpoint['iteration']
    return iteration


def sample_next_token(
    logits: torch.Tensor, temperature: float = 1.0, top_p: float = 1.0
) -> int:
    # logits:[vocab_size,]
    if temperature == 0:
        return torch.argmax(logits).item()
    sorted_logits, sorted_indices = torch.sort(
        logits, descending=True
    )  # 对 logits 进行排序，得到排序后的 logits 和对应的索引
    sorted_probs = softmax(sorted_logits / temperature, dim=-1)
    scaled_logits = sorted_logits / temperature
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    indice_to_remove = cumulative_probs > top_p
    indice_to_remove[1:] = indice_to_remove[
        :-1
    ].clone()  # 保留第一个超过 top_p 的 token
    indice_to_remove[0] = False
    scaled_logits[indice_to_remove] = float(
        '-inf'
    )  # 将不在 top_p 内的 token 的 logits 设置为 -inf
    probs = softmax(scaled_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)

    return sorted_indices[next_token].item()


def decode(
    model: transformer_lm,
    prompt_token_ids: list[int],
    end_token: int,
    ctx: int,
    device: str,
    max_generated: int = 32768,
    temperature: float = 1.0,
    top_p: float = 1.0,
):
    #generated_ = []
    tokens = prompt_token_ids[:]
    model.eval()
    with torch.no_grad():
        for _ in range(max_generated):
            x = torch.Tensor(tokens[-ctx:], dtype=torch.long, device=device).unsqueeze(0)
            logits = model(x)  # [1, seq, vocab]
            next_logits = logits[0, -1, :]

            next_token_id = sample_next_token(next_logits, temperature, top_p)
            #generated_.append(next_token_id)
            tokens.append(next_token_id)
            if next_token_id == end_token:
                break
    return tokens

if __name__ == "__main__":
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=1)

    for t in range(100):
        opt.zero_grad()  # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean()  # Compute a scalar loss value.
        print(loss.cpu().item())
        loss.backward()  # Run backward pass, which computes gradients.
        opt.step()  # Run optimizer step.
