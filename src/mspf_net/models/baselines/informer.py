from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import BaseBaselineModel, ensure_bcl


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 8192):
        super().__init__()
        self.d_model = d_model
        pe = self._build_pe(max_len)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def _build_pe(self, max_len: int) -> torch.Tensor:
        pe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def _ensure_length(self, seq_len: int, device: torch.device) -> None:
        if self.pe.size(1) >= seq_len:
            return
        new_len = max(seq_len, self.pe.size(1) * 2)
        pe = self._build_pe(new_len).unsqueeze(0).to(device=device)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_length(x.size(1), x.device)
        return x + self.pe[:, : x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, in_channels: int, d_model: int):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, d_model, kernel_size=3, padding=1, bias=False)
        self.norm = nn.BatchNorm1d(d_model)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        return x.transpose(1, 2).contiguous()


class ProbSparseSelfAttention(nn.Module):
    """
    Informer-style sparse attention approximation:
    select the most informative queries, compute exact attention for them,
    and keep a global context path for the rest.
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1, factor: int = 5):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.factor = factor

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        x = x.view(bsz, seq_len, self.n_heads, self.head_dim)
        return x.permute(0, 2, 1, 3).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self._reshape_heads(self.q_proj(x))
        k = self._reshape_heads(self.k_proj(x))
        v = self._reshape_heads(self.v_proj(x))

        scale = 1.0 / math.sqrt(self.head_dim)
        query_energy = q.pow(2).sum(dim=-1)
        top_u = min(seq_len, max(8, int(self.factor * math.log(seq_len + 1))))
        top_idx = torch.topk(query_energy, k=top_u, dim=-1).indices

        global_context = v.mean(dim=2, keepdim=True).expand(-1, -1, seq_len, -1).clone()

        q_top = torch.gather(q, 2, top_idx.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))
        scores = torch.matmul(q_top, k.transpose(-2, -1)) * scale
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        top_context = torch.matmul(attn, v)
        global_context.scatter_(
            2,
            top_idx.unsqueeze(-1).expand(-1, -1, -1, self.head_dim),
            top_context,
        )

        out = global_context.permute(0, 2, 1, 3).contiguous().view(bsz, seq_len, self.d_model)
        return self.out_proj(out)


class InformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, factor: int = 5):
        super().__init__()
        self.attn = ProbSparseSelfAttention(d_model, n_heads=n_heads, dropout=dropout, factor=factor)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ffn(x))
        return x


class ConvDistil(nn.Module):
    """
    Informer distilling block: halve temporal length between encoder stages.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm1d(d_model)
        self.act = nn.GELU()
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2).contiguous()
        x = self.pool(self.act(self.bn(self.conv(x))))
        return x.transpose(1, 2).contiguous()


class Informer(BaseBaselineModel):
    model_name = "informer"

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        factor: int = 5,
        distil: bool = True,
    ):
        super().__init__()
        self.embedding = TokenEmbedding(in_channels, d_model)
        self.positional = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [InformerEncoderLayer(d_model, n_heads=n_heads, dropout=dropout, factor=factor) for _ in range(n_layers)]
        )
        self.distillers = nn.ModuleList(
            [ConvDistil(d_model) for _ in range(max(0, n_layers - 1))]
        )
        self.use_distil = distil
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_bcl(x)
        x = self.embedding(x)
        x = self.dropout(self.positional(x))

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.use_distil and i < len(self.distillers):
                x = self.distillers[i](x)

        x = x.mean(dim=1)
        return self.head(x)
