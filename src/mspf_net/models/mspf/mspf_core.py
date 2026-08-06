"""MSPF-Net core: front-end, dual paths, heads, and MoE routing."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier

from .frontend import SharedFrontend
from .heads import ExpertRouter, HybridOutput, PathFusionGate
from .nonstationary import NonstationaryPath
from .time_frequency import TimeFrequencyPath
from .cnn_expert import CnnExpert, SpectralExpert


def flatten_mspf_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    """Merge nested YAML sections into flat constructor kwargs."""
    out = dict(cfg)
    for section in (
        "frontend",
        "periodic_path",
        "time_frequency_path",
        "nonstationary_path",
        "moe",
    ):
        nested = out.pop(section, None)
        if isinstance(nested, dict):
            out.update(nested)
    return out


class MSPFNetCore(nn.Module):
    model_name = "mspf_net"

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        architecture: str = "slim",
        stem_channels: int = 64,
        branch_channels: int = 128,
        feat_dim: int = 512,
        se_reduction: int = 16,
        use_se: bool = True,
        cwt_num_bands: int = 3,
        cwt_kernel_size: int = 129,
        use_early_channel_mixer: bool = True,
        channel_pooling: str = "attention",
        classifier_head: str = "softmax",
        dropout: float = 0.2,
        rf_n_estimators: int = 300,
        rf_max_depth: Optional[int] = None,
        rf_max_features: str | int | float = "sqrt",
        rf_min_samples_split: int = 2,
        rf_n_jobs: int = -1,
        use_sensor_fusion_stem: bool = False,
        sensor_fusion_mode: str = "residual_attention",
        use_fir_frontend: bool = False,
        fir_num_bands: int = 4,
        fir_kernel_size: int = 31,
        fir_init_mode: str = "delta",
        use_wavelet_residual: bool = False,
        wavelet_levels: int = 2,
        use_nonstationary_path: bool | None = None,
        scalogram_freq_bins: int = 32,
        scalogram_time_bins: int = 64,
        nonstationary_hidden_size: int = 128,
        nonstationary_num_layers: int = 1,
        nonstationary_downsample_stride: int = 4,
        path_fusion_mode: str = "gated_sum",
        min_periodic_gate: float = 0.2,
        max_periodic_gate: float = 0.95,
        router_hidden: int = 128,
        router_temperature: float = 1.0,
        router_entropy_weight: float = 0.0,
        moe_non_periodic_expert: str = "cnn",
        moe_disabled_experts: list[str] | None = None,
        moe_router_mode: str = "learned",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if kwargs:
            unknown = ", ".join(sorted(str(k) for k in kwargs))
            raise TypeError(f"MSPFNetCore got unexpected keyword argument(s): {unknown}")
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.feat_dim = int(feat_dim)
        self.classifier_head = str(classifier_head).lower()
        self.router_entropy_weight = float(router_entropy_weight)
        self._last_router_weights: torch.Tensor | None = None

        if self.classifier_head not in {"softmax", "rf"}:
            raise ValueError(f"classifier_head must be 'softmax' or 'rf', got {classifier_head!r}")
        self.architecture = str(architecture).lower()
        if self.architecture not in {"slim", "moe"}:
            raise ValueError(f"architecture must be 'slim' or 'moe', got {architecture!r}")
        if int(cwt_num_bands) < 1:
            raise ValueError(f"cwt_num_bands must be >= 1, got {cwt_num_bands!r}")
        if use_nonstationary_path is None:
            use_nonstationary_path = self.architecture == "slim"
        self.use_nonstationary_path = bool(use_nonstationary_path)

        self.moe_disabled_experts = {str(name) for name in (moe_disabled_experts or [])}
        self.moe_router_mode = str(moe_router_mode).lower()
        if self.moe_router_mode not in {"learned", "equal"}:
            raise ValueError(f"moe_router_mode must be 'learned' or 'equal', got {moe_router_mode!r}")

        self.frontend = SharedFrontend(
            in_channels=self.in_channels,
            stem_channels=stem_channels,
            use_early_channel_mixer=use_early_channel_mixer and not use_sensor_fusion_stem,
            use_sensor_fusion_stem=use_sensor_fusion_stem,
            sensor_fusion_mode=sensor_fusion_mode,
            use_fir_frontend=use_fir_frontend,
            fir_num_bands=fir_num_bands,
            fir_kernel_size=fir_kernel_size,
            fir_init_mode=fir_init_mode,
            use_wavelet_residual=use_wavelet_residual,
            wavelet_levels=wavelet_levels,
        )
        if self.architecture == "slim":
            self.periodic_path = TimeFrequencyPath(
                stem_channels=stem_channels,
                branch_channels=branch_channels,
                feat_dim=feat_dim,
                se_reduction=se_reduction,
                cwt_num_bands=cwt_num_bands,
                cwt_kernel_size=cwt_kernel_size,
                scalogram_freq_bins=scalogram_freq_bins,
                scalogram_time_bins=scalogram_time_bins,
                channel_pooling=channel_pooling,
                dropout=dropout,
                use_se=use_se,
            )
            self.moe_router = None
            self.moe_expert_names: list[str] = []
        else:
            non_periodic_kind = str(moe_non_periodic_expert).lower()
            if non_periodic_kind not in {"cnn", "spectral"}:
                raise ValueError(
                    f"moe_non_periodic_expert must be 'cnn' or 'spectral', got {moe_non_periodic_expert!r}"
                )
            self.moe_non_periodic_expert = non_periodic_kind
            self.moe_experts = nn.ModuleDict(
                {
                    "cwt": TimeFrequencyPath(
                        stem_channels=stem_channels,
                        branch_channels=branch_channels,
                        feat_dim=feat_dim,
                        se_reduction=se_reduction,
                        cwt_num_bands=cwt_num_bands,
                        cwt_kernel_size=cwt_kernel_size,
                        scalogram_freq_bins=scalogram_freq_bins,
                        scalogram_time_bins=scalogram_time_bins,
                        channel_pooling=channel_pooling,
                        dropout=dropout,
                        use_se=use_se,
                    ),
                    "non_periodic": (
                        CnnExpert(
                            stem_channels=stem_channels,
                            feat_dim=feat_dim,
                            se_reduction=se_reduction,
                            channel_pooling=channel_pooling,
                            dropout=dropout,
                        )
                        if non_periodic_kind == "cnn"
                        else SpectralExpert(
                            stem_channels=stem_channels,
                            branch_channels=branch_channels,
                            feat_dim=feat_dim,
                            cwt_num_bands=cwt_num_bands,
                            cwt_kernel_size=cwt_kernel_size,
                            se_reduction=se_reduction,
                            channel_pooling=channel_pooling,
                            dropout=dropout,
                        )
                    ),
                }
            )
            self.moe_expert_names = ["cwt", "non_periodic"]
            self.moe_router = ExpertRouter(
                feat_dim=feat_dim,
                num_experts=2,
                hidden=router_hidden,
                temperature=router_temperature,
            )
            self.periodic_path = None

        self.nonstationary_path = (
            NonstationaryPath(
                stem_channels=stem_channels,
                feat_dim=feat_dim,
                hidden_size=nonstationary_hidden_size,
                num_layers=nonstationary_num_layers,
                downsample_stride=nonstationary_downsample_stride,
                se_reduction=se_reduction,
                channel_pooling=channel_pooling,
                dropout=dropout,
            )
            if self.use_nonstationary_path
            else None
        )
        if self.architecture == "moe":
            self.path_gate = None
        else:
            self.path_gate = (
                PathFusionGate(
                    feat_dim=feat_dim,
                    path_fusion_mode=path_fusion_mode,
                    min_periodic_gate=min_periodic_gate,
                    max_periodic_gate=max_periodic_gate,
                )
                if self.use_nonstationary_path and self.nonstationary_path is not None
                else None
            )

        self.softmax_head = nn.Linear(self.feat_dim, self.num_classes)
        self.rf_cfg = {
            "n_estimators": int(rf_n_estimators),
            "max_depth": rf_max_depth,
            "max_features": rf_max_features,
            "min_samples_split": int(rf_min_samples_split),
            "n_jobs": int(rf_n_jobs),
        }
        self.rf_classifier: RandomForestClassifier | None = None
        self._last_periodic_gate: float | None = None

    def _moe_expert_mask(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        mask = torch.ones(len(self.moe_expert_names), device=device, dtype=dtype)
        for idx, name in enumerate(self.moe_expert_names):
            if name in self.moe_disabled_experts:
                mask[idx] = 0.0
        if float(mask.sum()) <= 0.0:
            raise ValueError(
                f"All MoE experts disabled via moe_disabled_experts={sorted(self.moe_disabled_experts)!r}"
            )
        return mask

    def _moe_weights(
        self,
        feats: list[torch.Tensor],
        confs: list[float | torch.Tensor],
    ) -> torch.Tensor:
        batch_size = int(feats[0].shape[0])
        device = feats[0].device
        dtype = feats[0].dtype
        mask = self._moe_expert_mask(batch_size, device, dtype)

        if self.moe_router_mode == "equal":
            weights = mask / mask.sum().clamp_min(1e-8)
            return weights.unsqueeze(0).expand(batch_size, -1)

        assert self.moe_router is not None
        weights = self.moe_router(feats, confs)
        weights = weights * mask.unsqueeze(0)
        return weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        feat_bc, batch_size, channels = self.frontend(x)
        if self.architecture == "moe":
            assert hasattr(self, "moe_expert_names")

            feats: list[torch.Tensor] = []
            confs: list[float | torch.Tensor] = []

            for name in self.moe_expert_names:
                expert = self.moe_experts[name]
                feat, conf = expert(feat_bc, batch_size, channels)
                feats.append(feat)
                confs.append(conf)

            weights = self._moe_weights(feats, confs)
            self._last_router_weights = weights.detach()

            fused = torch.zeros_like(feats[0])
            for i, feat in enumerate(feats):
                fused = fused + weights[:, i].unsqueeze(-1) * feat
            return fused

        periodic_feat, confidence = self.periodic_path(feat_bc, batch_size, channels)
        if self.nonstationary_path is None or self.path_gate is None:
            return periodic_feat
        nonstat_feat = self.nonstationary_path(feat_bc, batch_size, channels)
        fused, gate = self.path_gate(periodic_feat, nonstat_feat, confidence)
        self._last_periodic_gate = gate
        return fused

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softmax_head(self.extract_features(x))

    def compute_losses(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        domain_id: torch.Tensor | None = None,
        cls_criterion: nn.Module | None = None,
        x_aug: torch.Tensor | None = None,
    ) -> HybridOutput:
        del domain_id, x_aug
        feat = self.extract_features(x)
        logits = self.softmax_head(feat)
        aux: dict[str, torch.Tensor] = {}
        metrics: dict[str, float] = {}

        if cls_criterion is not None:
            aux["cls"] = cls_criterion(logits, y)
        else:
            aux["cls"] = F.cross_entropy(logits, y)

        if self._last_periodic_gate is not None:
            metrics["periodic_gate"] = float(self._last_periodic_gate)

        if self._last_router_weights is not None:
            w = self._last_router_weights
            metrics["router_entropy"] = float((-(w * (w + 1e-8).log()).sum(dim=1)).mean().item())
            for idx, name in enumerate(getattr(self, "moe_expert_names", [])):
                metrics[f"router_w_{name}"] = float(w[:, idx].mean().item())

        return HybridOutput(logits=logits, aux_losses=aux, metrics=metrics)

    def fit_random_forest(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        random_state: int = 42,
    ) -> None:
        self.rf_classifier = RandomForestClassifier(
            criterion="gini",
            random_state=int(random_state),
            **self.rf_cfg,
        )
        self.rf_classifier.fit(features, labels)

    def predict_with_rf(self, features: np.ndarray) -> np.ndarray:
        if self.rf_classifier is None:
            raise RuntimeError("Random Forest not fitted. Call fit_random_forest() first.")
        return self.rf_classifier.predict(features)

    def predict_proba_with_rf(self, features: np.ndarray) -> np.ndarray:
        if self.rf_classifier is None:
            raise RuntimeError("Random Forest not fitted. Call fit_random_forest() first.")
        return self.rf_classifier.predict_proba(features)

    def num_parameters(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))
