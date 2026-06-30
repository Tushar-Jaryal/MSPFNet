"""Classification heads, path fusion, and MoE routing."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class HybridOutput:
    logits: torch.Tensor
    aux_losses: dict[str, torch.Tensor] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)


class PathFusionGate(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        path_fusion_mode: str = "gated_sum",
        min_periodic_gate: float = 0.2,
        max_periodic_gate: float = 0.95,
    ) -> None:
        super().__init__()
        self.path_fusion_mode = str(path_fusion_mode).lower()
        self.min_periodic_gate = float(min_periodic_gate)
        self.max_periodic_gate = float(max_periodic_gate)
        self.gate_fc = nn.Sequential(
            nn.Linear(feat_dim * 2 + 3, feat_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim // 4, 1),
        )

    def forward(
        self,
        periodic_feat: torch.Tensor,
        nonstat_feat: torch.Tensor,
        periodic_confidence: float,
    ) -> tuple[torch.Tensor, float]:
        conf = torch.tensor([periodic_confidence], device=periodic_feat.device, dtype=periodic_feat.dtype)
        summary = torch.cat([periodic_feat.mean(dim=-1, keepdim=True), nonstat_feat.mean(dim=-1, keepdim=True)], dim=-1)
        gate_in = torch.cat([periodic_feat, nonstat_feat, summary, conf.expand(periodic_feat.size(0), 1)], dim=-1)
        learned = torch.sigmoid(self.gate_fc(gate_in)).squeeze(-1)
        base = self.min_periodic_gate + (self.max_periodic_gate - self.min_periodic_gate) * float(periodic_confidence)
        periodic_gate = (base * learned + (1.0 - learned) * self.min_periodic_gate).clamp(
            self.min_periodic_gate, self.max_periodic_gate
        )
        if self.path_fusion_mode == "equal":
            periodic_gate = torch.full_like(periodic_gate, 0.5)
        fused = periodic_gate.unsqueeze(-1) * periodic_feat + (1.0 - periodic_gate.unsqueeze(-1)) * nonstat_feat
        return fused, float(periodic_gate.mean().item())


class ExpertRouter(nn.Module):
    """Per-sample soft routing over multiple experts (mixture-of-experts)."""

    def __init__(
        self,
        feat_dim: int,
        num_experts: int,
        hidden: int = 128,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.num_experts = int(num_experts)
        self.hidden = int(hidden)
        self.temperature = float(temperature)

        if self.num_experts < 2:
            raise ValueError(f"num_experts must be >= 2, got {num_experts}")

        num_pairs = (self.num_experts * (self.num_experts - 1)) // 2
        in_dim = self.num_experts * 2 + num_pairs
        self.net = nn.Sequential(
            nn.Linear(in_dim, self.hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden, self.num_experts),
        )

    def set_temperature(self, value: float) -> None:
        self.temperature = float(value)

    def forward(
        self,
        expert_feats: list[torch.Tensor],
        expert_confs: list[float | torch.Tensor],
    ) -> torch.Tensor:
        if len(expert_feats) != self.num_experts:
            raise ValueError(f"Expected {self.num_experts} expert feats, got {len(expert_feats)}")
        if len(expert_confs) != self.num_experts:
            raise ValueError(f"Expected {self.num_experts} expert confs, got {len(expert_confs)}")

        b = int(expert_feats[0].shape[0])
        device = expert_feats[0].device
        dtype = expert_feats[0].dtype
        for feat in expert_feats:
            if feat.shape != (b, self.feat_dim):
                raise ValueError(f"Expected expert feat shape (B,{self.feat_dim}), got {tuple(feat.shape)}")

        conf_tensors: list[torch.Tensor] = []
        for conf in expert_confs:
            if isinstance(conf, torch.Tensor):
                c = conf.to(device=device, dtype=dtype)
                if c.ndim == 0:
                    c = c.expand(b)
                elif c.ndim == 1 and c.shape[0] == 1:
                    c = c.expand(b)
                elif c.ndim != 1 or c.shape[0] != b:
                    raise ValueError(f"Expected conf tensor shape (B,) or scalar, got {tuple(c.shape)}")
                conf_tensors.append(c)
            else:
                conf_tensors.append(torch.full((b,), float(conf), device=device, dtype=dtype))

        norms = [feat.norm(p=2, dim=1).clamp_min(1e-8) for feat in expert_feats]
        confs = [c.clamp(0.0, 1.0) for c in conf_tensors]

        pairwise_cos: list[torch.Tensor] = []
        for i in range(self.num_experts):
            fi = expert_feats[i]
            ni = norms[i]
            for j in range(i + 1, self.num_experts):
                fj = expert_feats[j]
                nj = norms[j]
                cos = (fi * fj).sum(dim=1) / (ni * nj)
                pairwise_cos.append(cos.clamp(-1.0, 1.0))

        router_in = torch.stack(confs + norms + pairwise_cos, dim=1)
        logits = self.net(router_in)
        temp = max(float(self.temperature), 1e-3)
        return torch.softmax(logits / temp, dim=1)
