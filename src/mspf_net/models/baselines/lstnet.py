from __future__ import annotations

import torch
import torch.nn as nn

from .common import BaseBaselineModel, ensure_bcl


class LSTNet(BaseBaselineModel):
    """
    LSTNet-style classifier:
    - temporal CNN front-end for local patterns
    - main GRU for long-range dependencies
    - skip GRU for periodic structure
    - highway path over the recent raw signal
    """

    model_name = "lstnet"

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        width: int = 64,
        hidden_size: int = 128,
        dropout: float = 0.1,
        kernel_size: int = 6,
        skip: int = 16,
        skip_hidden_size: int | None = None,
        highway_window: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.width = width
        self.hidden_size = hidden_size
        self.skip = skip
        self.highway_window = highway_window
        self.skip_hidden_size = skip_hidden_size or max(hidden_size // 2, 32)

        self.conv = nn.Conv1d(in_channels, width, kernel_size=kernel_size)
        self.conv_act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        self.gru = nn.GRU(
            input_size=width,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.skip_gru = nn.GRU(
            input_size=width,
            hidden_size=self.skip_hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.highway = nn.Linear(in_channels * highway_window, hidden_size)

        fusion_dim = hidden_size
        if skip > 0:
            fusion_dim += skip * self.skip_hidden_size
        if highway_window > 0:
            fusion_dim += hidden_size

        self.head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes),
        )

    def _skip_path(self, conv_seq: torch.Tensor) -> torch.Tensor | None:
        if self.skip <= 0:
            return None

        batch, steps, channels = conv_seq.shape
        period_count = steps // self.skip
        if period_count <= 0:
            return None

        x = conv_seq[:, -period_count * self.skip :, :]
        x = x.view(batch, period_count, self.skip, channels)
        x = x.permute(0, 2, 1, 3).contiguous().view(batch * self.skip, period_count, channels)
        _, h = self.skip_gru(x)
        h = h[-1].view(batch, self.skip * self.skip_hidden_size)
        return h

    def _highway_path(self, raw_x: torch.Tensor) -> torch.Tensor | None:
        if self.highway_window <= 0:
            return None

        window = min(self.highway_window, raw_x.size(-1))
        x = raw_x[:, :, -window:]
        if window < self.highway_window:
            pad = self.highway_window - window
            x = nn.functional.pad(x, (pad, 0))
        x = x.reshape(x.size(0), self.in_channels * self.highway_window)
        return self.highway(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_bcl(x)
        raw_x = x

        conv = self.conv_act(self.conv(x))
        conv = self.dropout(conv)
        conv_seq = conv.transpose(1, 2).contiguous()

        _, h = self.gru(conv_seq)
        features = [h[-1]]

        skip_feat = self._skip_path(conv_seq)
        if skip_feat is not None:
            features.append(self.dropout(skip_feat))

        highway_feat = self._highway_path(raw_x)
        if highway_feat is not None:
            features.append(self.dropout(highway_feat))

        fused = torch.cat(features, dim=1)
        return self.head(fused)
