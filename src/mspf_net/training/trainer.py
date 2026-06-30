from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from mspf_net.training.eval_utils import evaluate_classifier, evaluate_noise_robustness
from mspf_net.training.metrics import classification_metrics
from mspf_net.utils.device import (
    cuda_peak_memory_mb,
    device_info,
    process_rss_mb,
    reset_cuda_peak_memory,
)


def _to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        output_dir: Path,
        run_name: str,
        class_names: list[str],
        max_epochs: int = 100,
        patience: int = 20,
        grad_clip_norm: float | None = None,
        use_amp: bool = False,
        accum_steps: int = 1,
        selection_metric: str = "val_macro_f1",
        batch_size: int | None = None,
    ):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.class_names = class_names
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.grad_clip_norm = None if grad_clip_norm is None else float(grad_clip_norm)
        self.use_amp = bool(use_amp) and self.device.type == "cuda"
        self.accum_steps = max(int(accum_steps), 1)
        self.selection_metric = str(selection_metric)
        self.batch_size = None if batch_size is None else int(batch_size)
        self.scaler = GradScaler("cuda", enabled=self.use_amp)

        self.ckpt_dir = self.output_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.best_ckpt = self.ckpt_dir / f"{run_name}_best.pt"
        self.history_path = self.output_dir / f"{run_name}_history.csv"
        self.curves_path = self.output_dir / f"{run_name}_curves.png"

    def _unpack_batch(self, batch):
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            return batch[0], batch[1]
        raise ValueError(f"Expected batch of (x, y), got {type(batch)!r}")

    def _forward_loss(self, x, y):
        with autocast(device_type="cuda", enabled=self.use_amp):
            logits = self.model(x)
            loss = self.criterion(logits, y)
        return loss, logits

    def _run_epoch(self, loader, train: bool) -> tuple[float, float]:
        self.model.train(train)
        total_loss = 0.0
        total_correct = 0
        total_count = 0
        if train:
            self.optimizer.zero_grad(set_to_none=True)
        for step_idx, batch in enumerate(loader, start=1):
            x, y = self._unpack_batch(batch)
            x = x.to(self.device, dtype=torch.float32, non_blocking=True)
            y = y.to(self.device, dtype=torch.long, non_blocking=True)
            loss, logits = self._forward_loss(x, y)
            if train:
                scaled_loss = loss / self.accum_steps
                self.scaler.scale(scaled_loss).backward()
                should_step = (step_idx % self.accum_steps == 0) or (step_idx == len(loader))
                if should_step:
                    if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.item()) * len(x)
            total_correct += int((logits.argmax(dim=1) == y).sum().item())
            total_count += int(len(x))
        total_count = max(total_count, 1)
        return total_loss / total_count, total_correct / total_count

    @torch.no_grad()
    def evaluate(self, loader) -> dict:
        return evaluate_classifier(
            self.model,
            loader,
            self.device,
            self.class_names,
            criterion=self.criterion,
            use_amp=self.use_amp,
        )

    @torch.no_grad()
    def inference_ms_per_window(self, loader, n_batches: int = 10) -> float:
        self.model.eval()
        batches = []
        for i, batch in enumerate(loader):
            batches.append(batch)
            if i + 1 >= n_batches:
                break
        if not batches:
            return float("nan")
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        n = 0
        for batch in batches:
            x, _ = self._unpack_batch(batch)
            x = x.to(self.device, dtype=torch.float32, non_blocking=True)
            with autocast(device_type="cuda", enabled=self.use_amp):
                _ = self.model(x)
            n += len(x)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        return (elapsed / max(n, 1)) * 1000.0

    def _save_curves(self, history: pd.DataFrame) -> None:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(history["epoch"], history["train_loss"], label="train")
        axes[0].plot(history["epoch"], history["val_loss"], label="val")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()
        axes[1].plot(history["epoch"], history["train_acc"], label="train_acc")
        axes[1].plot(history["epoch"], history["val_macro_f1"], label="val_macro_f1")
        axes[1].set_title("Training Dynamics")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(self.curves_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    def fit(self, train_loader, val_loader, test_loader=None, robustness_loader=None, evaluation_split: str = "test") -> dict:
        print(f"  Training {self.run_name} — {self.max_epochs} epochs")
        print(f"  Device: {self.device}")
        print(f"  Early stop: patience={self.patience}")
        print(f"  Selection metric: {self.selection_metric}")
        if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
            print(f"  Grad clip: max_norm={self.grad_clip_norm:g}")
        if self.use_amp:
            print("  AMP: enabled")
        if self.accum_steps > 1:
            print(f"  Gradient accumulation: {self.accum_steps} steps")
        if self.batch_size is not None:
            effective = self.batch_size * self.accum_steps
            print(f"  Batch size: {self.batch_size}  (effective={effective})")

        reset_cuda_peak_memory(self.device)
        history_rows = []
        best_epoch = 0
        best_f1 = -1.0
        wait = 0
        start_time = time.perf_counter()

        for epoch in range(1, self.max_epochs + 1):
            train_loss, train_acc = self._run_epoch(train_loader, train=True)
            val = self.evaluate(val_loader)
            lr = float(self.optimizer.param_groups[0]["lr"])
            selected_score = self._selection_metric_value(val)
            improved = selected_score > best_f1
            if improved:
                best_f1 = selected_score
                best_epoch = epoch
                wait = 0
                torch.save(
                    {
                        "model_state": self.model.state_dict(),
                        "best_epoch": best_epoch,
                        "best_val_metric": best_f1,
                        "selection_metric": self.selection_metric,
                        "class_names": self.class_names,
                    },
                    self.best_ckpt,
                )
            else:
                wait += 1

            row = {
                "epoch": epoch,
                "batch_size": self.batch_size,
                "peak_gpu_mb": cuda_peak_memory_mb(self.device),
                "rss_mb": process_rss_mb(),
                "train_loss": train_loss,
                "train_acc": train_acc * 100.0,
                "val_loss": val["loss"],
                "val_accuracy": val["accuracy"],
                "val_macro_f1": val["macro_f1"],
                "val_macro_precision": val["macro_precision"],
                "val_macro_recall": val["macro_recall"],
                "val_file_macro_f1": (val.get("file_level") or {}).get("macro_f1"),
                "lr": lr,
                "best_epoch_so_far": best_epoch,
                "selection_metric_value": selected_score,
            }
            history_rows.append(row)
            status = "best=updated" if improved else ""
            print(
                f"  Epoch {epoch:4d}/{self.max_epochs}  "
                f"tr_loss={train_loss:.4f}  tr_acc={train_acc * 100.0:.1f}%  "
                f"val Acc={val['accuracy']:.2f}%  F1={val['macro_f1']:.2f}%  "
                f"Prec={val['macro_precision']:.2f}%  Rec={val['macro_recall']:.2f}%  "
                f"lr={lr:.2e}  {status}"
            )
            self.scheduler.step()

            if wait >= self.patience:
                print(f"  Early stop at epoch {epoch}")
                break

        elapsed = time.perf_counter() - start_time
        history = pd.DataFrame(history_rows)
        history.to_csv(self.history_path, index=False)
        self._save_curves(history)

        ckpt = torch.load(self.best_ckpt, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])

        peak_train_gpu_mb = cuda_peak_memory_mb(self.device)
        reset_cuda_peak_memory(self.device)
        evaluated = self.evaluate(test_loader) if test_loader is not None else None
        robustness = self.evaluate(robustness_loader) if robustness_loader is not None else None
        noise_robustness = (
            evaluate_noise_robustness(
                self.model,
                test_loader,
                self.device,
                self.class_names,
                use_amp=self.use_amp,
            )
            if test_loader is not None
            else None
        )
        infer_ms = self.inference_ms_per_window(test_loader if test_loader is not None else val_loader)
        peak_inference_gpu_mb = cuda_peak_memory_mb(self.device)
        effective_batch_size = (
            None if self.batch_size is None else int(self.batch_size * self.accum_steps)
        )
        memory = {
            "peak_train_gpu_mb": peak_train_gpu_mb,
            "peak_inference_gpu_mb": peak_inference_gpu_mb,
            "peak_rss_mb": process_rss_mb(),
            "device_info": device_info(self.device),
        }

        result = {
            "run_name": self.run_name,
            "best_epoch": int(best_epoch),
            "epochs_ran": int(len(history)),
            "elapsed_s": float(elapsed),
            "best_checkpoint": str(self.best_ckpt),
            "history_csv": str(self.history_path),
            "curves_png": str(self.curves_path),
            "val": None if val is None else _to_jsonable(val),
            evaluation_split: None if evaluated is None else _to_jsonable(evaluated),
            "robustness": None if robustness is None else _to_jsonable(robustness),
            "noise_robustness": None if noise_robustness is None else _to_jsonable(noise_robustness),
            "inference_ms_per_window": float(infer_ms),
            "selection_metric": self.selection_metric,
            "batch_size": self.batch_size,
            "effective_batch_size": effective_batch_size,
            "memory": memory,
        }
        if evaluation_split != "test" and "test" not in result:
            result["test"] = None
        return result

    def _selection_metric_value(self, val: dict) -> float:
        if self.selection_metric == "val_file_macro_f1":
            return float((val.get("file_level") or {}).get("macro_f1", float("-inf")))
        return float(val.get("macro_f1", float("-inf")))
