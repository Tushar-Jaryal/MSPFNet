from __future__ import annotations

import math

import torch
import torch.nn as nn

from .common import BaseBaselineModel, ensure_bcl


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.d_model = d_model
        self.register_buffer("pe", self._build_pe(max_len), persistent=False)

    def _build_pe(self, max_len: int) -> torch.Tensor:
        pe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.pe.size(1):
            self.pe = self._build_pe(x.size(1)).to(device=x.device, dtype=x.dtype)
        return x + self.pe[:, : x.size(1)]


class Transformer1D(BaseBaselineModel):
    """
    Patch-tokenized Transformer encoder for vibration classification.

    Matches the plan structure closely:
    - patch tokenizer with patch_size=stride=16
    - 4-layer encoder
    - 8 attention heads
    - d_model=256 by default
    """

    model_name = "transformer1d"

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        patch_size: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_embed = nn.Conv1d(
            in_channels,
            d_model,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        # 8192-sample windows with patch_size=16 create 512 patch tokens, plus one CLS token.
        # Keep a safe default and let the encoding expand automatically for larger sequences.
        self.positional = PositionalEncoding(d_model, max_len=513)
        self.dropout = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_bcl(x)
        x = self.patch_embed(x).transpose(1, 2).contiguous()
        bsz = x.size(0)
        cls = self.cls_token.expand(bsz, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.dropout(self.positional(x))
        x = self.encoder(x)
        x = self.norm(x[:, 0])
        return self.head(x)
