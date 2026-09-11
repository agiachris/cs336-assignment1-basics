from typing import Optional, Iterator, List

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


def init_weight_matrix(
    in_features: int, 
    out_features: int,
    device: Optional[torch.device] = None, 
    dtype: Optional[torch.dtype] = None 
) -> torch.Tensor:
    std = math.sqrt(2 / (in_features + out_features))
    weight = torch.empty((out_features, in_features), dtype=dtype, device=device)
    torch.nn.init.trunc_normal_(weight, mean=0, std=std, a=-3*std, b=3*std)
    return weight


def init_embedding_matrix(
    num_embeddings: int, 
    embedding_dim: int, 
    device: Optional[torch.device] = None, 
    dtype: Optional[torch.dtype] = None 
) -> torch.Tensor:
    weight = torch.empty((num_embeddings, embedding_dim), dtype=dtype, device=device)
    torch.nn.init.trunc_normal_(weight, mean=0, std=1, a=-3, b=3)
    return weight


def init_vec(
    d_model: int,
    device: Optional[torch.device] = None, 
    dtype: Optional[torch.dtype] = None 
) -> torch.Tensor:
    return torch.ones(d_model, device=device, dtype=dtype)


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x = x - x.amax(dim=dim, keepdim=True)
    exp = torch.exp(x)
    return exp / exp.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.BoolTensor] = None,
) -> torch.Tensor:
    attn_weights = torch.einsum("...sd, ...td -> ...st", Q, K) / math.sqrt(K.shape[-1])
    if mask is not None:
        attn_weights = attn_weights.masked_fill(~mask, float("-inf"))
    attn_weights = softmax(attn_weights, dim=-1)
    return torch.einsum("...st, ...td -> ...sd", attn_weights, V)


def cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor
) -> torch.Tensor:
    logits_BSV = logits
    targets_BS = targets
    logits_tgt_BS = logits_BSV.take_along_dim(targets_BS.unsqueeze(-1), dim=-1).squeeze(-1)
    loss = (torch.logsumexp(logits_BSV, dim=-1) - logits_tgt_BS).mean()
    return loss


def perplexity(
    logits: torch.Tensor,
    targets: torch.Tensor
) -> torch.Tensor:
    logits_BSV = logits
    targets_BS = targets
    logits_tgt_BS = logits_BSV.take_along_dim(targets_BS.unsqueeze(-1), dim=-1).squeeze(-1)
    nll_BS = torch.logsumexp(logits_BSV, dim=-1) - logits_tgt_BS
    perplexity_B = torch.exp(nll_BS.mean(dim=-1))
    return perplexity_B


def gradient_clipping(
    params: List[nn.Parameter], 
    m: float, 
    eps: float = 1e-6
) -> None:

    l2 = 0
    for p in params:
        if p.grad is None:
            continue
        l2 += torch.sum(p.grad.data ** 2)
    l2 = torch.sqrt(l2)

    if l2.item() > m:
        mod = m / (l2 + eps)

        for p in params:
            if p.grad is None:
                continue
            p.grad.mul_(mod)
    

class Linear(nn.Module):

    def __init__(
        self,
        in_features: int, 
        out_features: int, 
        device: Optional[torch.device] = None, 
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        weight = init_weight_matrix(
            in_features=in_features,
            out_features=out_features,
            device=device,
            dtype=dtype,
        )
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute forward pass.
        
        Args:
            x: torch.Tensor (B, in_features)

        Returns:
            torch.Tensor (B, out_features)
        """
        x_BD = x
        return x_BD @ self.weight.T


class Embedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int, 
        embedding_dim: int, 
        device: Optional[torch.device] = None, 
        dtype: Optional[torch.dtype] = None 
    ):
        super().__init__()
        weight = init_embedding_matrix(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            device=device,
            dtype=dtype,
        )
        self.weight = nn.Parameter(weight)
        self.d_model = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ids_BS = x
        embeddings_BSE = self.weight[ids_BS.reshape(-1)].reshape((ids_BS.shape[0], ids_BS.shape[1], self.d_model))
        return embeddings_BSE


class RMSNorm(nn.Module):

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: Optional[torch.device] = None, 
        dtype: Optional[torch.dtype] = None 
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.g_E = nn.Parameter(init_vec(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x_BSE = x.to(torch.float32)
        rms_BS1 = torch.sqrt((x_BSE ** 2).mean(dim=-1, keepdim=True) + self.eps)
        rms_norm_BSE = x_BSE / rms_BS1 * self.g_E
        return rms_norm_BSE.to(in_dtype)


class SwiGLUFFN(nn.Module):

    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        device: Optional[torch.device] = None, 
        dtype: Optional[torch.dtype] = None 
    ):
        super().__init__()
        def nearest_64(v: float) -> int:
            ranges = (np.arange(v // 32) + 1) * 64
            diff = (ranges - v) ** 2
            return ranges[diff.argmin()]
            
        d_ff = d_ff if isinstance(d_ff, int) else nearest_64(d_model * 8 / 3)

        self.weight_1 = nn.Parameter(
            init_weight_matrix(
                in_features=d_model,
                out_features=d_ff,
                device=device,
                dtype=dtype,
            )
        )

        self.weight_3 = nn.Parameter(
            init_weight_matrix(
                in_features=d_model,
                out_features=d_ff,
                device=device,
                dtype=dtype,
            )
        )

        self.weight_2 = nn.Parameter(
            init_weight_matrix(
                in_features=d_ff,
                out_features=d_model,
                device=device,
                dtype=dtype,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z1 = x @ self.weight_1.T
        x1 = z1 * torch.sigmoid(z1)
        x3 = x @ self.weight_3.T
        return (x1 * x3) @ self.weight_2.T


class RoPE(nn.Module):
 
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None 
    ):
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even"
        self.d_k = d_k
 
        angle_row_S = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        angle_col_D = theta ** ((2 * (1 + torch.arange(d_k // 2, device=device, dtype=torch.float32)) - 2) / d_k)
        angle_mat_SD = angle_row_S[:, None] / angle_col_D[None, :]
        cos_SD = torch.cos(angle_mat_SD)
        sin_SD = torch.sin(angle_mat_SD)
        rope_SD22 = torch.stack((cos_SD, -sin_SD, sin_SD, cos_SD), dim=-1).reshape((max_seq_len, d_k // 2, 2, 2)).to(dtype)
        self.register_buffer("rope", rope_SD22, persistent=False)
 
    def forward(
        self,
        x: torch.Tensor,
        token_positions: Optional[torch.Tensor]
    ) -> torch.Tensor:
        in_dims = tuple(x.shape)
        x_dimsTD2 = x.reshape(in_dims[:-1] + (in_dims[-1] // 2, 2))
 
        rope_TD22: torch.Tensor = self.rope[token_positions]
        while rope_TD22.ndim < x_dimsTD2.ndim + 1:
            rope_TD22 = rope_TD22.unsqueeze(0)
 
        y_dimsTD2 = torch.einsum("...ij,...j->...i", rope_TD22.to(x.dtype), x_dimsTD2)
        return y_dimsTD2.reshape(in_dims)


class CausalMultiHeadAttention(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float = 10000,
        max_seq_len: int = 10000,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None 
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.device = device

        # Linear layers.
        self.Wq = Linear(d_model, d_model, device=device, dtype=dtype)
        self.Wv = Linear(d_model, d_model, device=device, dtype=dtype)
        self.Wk = Linear(d_model, d_model, device=device, dtype=dtype)
        self.Wo = Linear(d_model, d_model, device=device, dtype=dtype)

        # RoPE embedding.
        self.rope = RoPE(
            theta=theta,
            d_k = self.head_dim,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype
        )

        # Mask buffer (might need to create online.)
        mask = torch.tril(torch.ones((max_seq_len, max_seq_len), device=device)).to(torch.bool)
        self.register_buffer("causal_mask", mask, persistent=False)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape

        # Step 1. Build query, key, value vectors.
        Q_BHSD = self.Wq(x).reshape((B, S, self.num_heads, self.head_dim)).permute((0, 2, 1, 3))
        K_BHSD = self.Wk(x).reshape((B, S, self.num_heads, self.head_dim)).permute((0, 2, 1, 3))
        V_BHSD = self.Wv(x).reshape((B, S, self.num_heads, self.head_dim)).permute((0, 2, 1, 3))

        # Step 2. Multi-head self attention with causal mask.
        O_BHSD = scaled_dot_product_attention(
            self.rope(Q_BHSD, token_positions=token_positions), 
            self.rope(K_BHSD, token_positions=token_positions), 
            V_BHSD, 
            mask=self.causal_mask[:S, :S].to(x.device))

        # Step 3. Perform output projection.
        O_BSD = self.Wo(O_BHSD.permute((0, 2, 1, 3)).reshape(B, S, -1))

        return O_BSD


class TransformerBlock(nn.Module):

    def __init__(
        self,
        # CausalMultiHeadAttention params.
        d_model: int,
        num_heads: int,
        theta: float = 10000,
        max_seq_len: int = 10000,
        # SwiGLUFFN params.
        d_ff: Optional[int] = None,
        # RMSNorm params.
        eps: float = 1e-5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.attn_layer = CausalMultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            theta=theta,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
        )
        self.ffnn_layer = SwiGLUFFN(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype,
        )

        self.attn_norm = RMSNorm(
            d_model=d_model,
            eps=eps,
            device=device,
            dtype=dtype
        )

        self.ffnn_norm = RMSNorm(
            d_model=d_model,
            eps=eps,
            device=device,
            dtype=dtype
        )

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor
    ) -> torch.Tensor:
        x = x + self.attn_layer(self.attn_norm(x), token_positions)
        x = x + self.ffnn_layer(self.ffnn_norm(x))
        return x


class TransformerLLM(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        theta: float = 10000,
        d_ff: Optional[int] = None,
        eps: float = 1e-5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()

        # Step 1. Create embedding layer.
        self.embedding = Embedding(
            num_embeddings=vocab_size, 
            embedding_dim=d_model,
            device=device,
            dtype=dtype
        )

        # Step 2. Create attention layers.
        attn_layers = []
        for _ in range(num_layers):
            attn_layers.append(
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    theta=theta,
                    max_seq_len=context_length,
                    d_ff=d_ff,
                    eps=eps,
                    device=device,
                    dtype=dtype
                )
            )
        self.attn_layers = nn.ModuleList(attn_layers)
        self.attn_norm = RMSNorm(
            d_model=d_model,
            eps=eps,
            device=device,
            dtype=dtype
        )

        # Step 3. Create output projection. 
        self.proj_out = Linear(
            in_features=d_model, 
            out_features=vocab_size,
            device=device,
            dtype=dtype
        )

    def forward(
        self, 
        x: torch.Tensor, 
        token_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Step 0. Get token_positions.
        B, S = x.shape
        if token_positions is None:
            token_positions = torch.arange(S, device=x.device)

        # Step 1. Embedding layer.
        z = self.embedding(x)
        
        # Step 2. Attenion blocks.
        for block in self.attn_layers:
            z = block(z, token_positions=token_positions)
        z = self.attn_norm(z)

        # Step 3. Output projection.
        return self.proj_out(z)


class SGD(torch.optim.Optimizer):

    def __init__(
        self,
        params: Iterator[nn.Parameter],
        lr: float = 1e-3, 
    ):
        defaults = {"lr": lr}
        super().__init__(params=params, defaults=defaults)

    @torch.no_grad
    def step(self, closure = None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0)
                p.data -= lr / math.sqrt(t + 1) * p.grad
                state["t"] = t + 1 

        return loss


class AdamW(torch.optim.Optimizer):

    def __init__(
        self,
        params: Iterator[nn.Parameter],
        lr: float = 1e-3, 
        beta1: float = 0.9,
        beta2: float = 0.999,
        weight_decay: float = 1e-5,
        eps: float = 1e-8
    ):
        defaults = {
            "lr": lr, 
            "beta1": beta1,
            "beta2": beta2,
            "weight_decay": weight_decay,
            "eps": eps
        }
        super().__init__(params=params, defaults=defaults)

    @torch.no_grad
    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 1
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                # Weight decay.
                p.data -= lr * weight_decay * p.data
                
                # Gradient update.
                t = state["t"]
                lr_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                m = beta1 * state["m"] + (1 - beta1) * p.grad
                v = beta2 * state["v"] + (1 - beta2) * (p.grad ** 2)
                p.data -= lr_t * m / (torch.sqrt(v) + eps)

                state["t"] = t + 1
                state["m"] = m
                state["v"] = v

        return loss
