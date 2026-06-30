import os
import warnings
from typing import Union, Optional

import torch
import torch.nn as nn
import numpy as np

# ─── MPS environment setup ────────────────────────────────────────────────────
# Allow unsupported MPS ops to fall back to CPU silently rather than crash.
# Set BEFORE any torch calls. Safe to set on all platforms.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


# ─── Device resolution ────────────────────────────────────────────────────────

def get_device(requested: Optional[str] = "auto") -> torch.device:
    """
    Resolve the best available device.

    Args:
        requested: "auto" | "mps" | "cuda" | "cpu"
            "auto" → MPS if Apple Silicon, CUDA if NVIDIA GPU, else CPU

    Returns:
        torch.device

    Examples:
        >>> device = get_device()          # auto
        >>> device = get_device("mps")     # force MPS (raises if unavailable)
        >>> device = get_device("cpu")     # force CPU
    """
    requested = "auto" if requested is None else requested.lower().strip()

    if requested == "auto":
        if _mps_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    elif requested == "mps":
        if not _mps_available():
            warnings.warn(
                "MPS requested but not available. "
                "Check: torch.backends.mps.is_available(). "
                "Falling back to CPU.",
                RuntimeWarning,
                stacklevel=2,
            )
            return torch.device("cpu")
        return torch.device("mps")

    elif requested == "cuda":
        if not torch.cuda.is_available():
            warnings.warn(
                "CUDA requested but not available. Falling back to CPU.",
                RuntimeWarning,
                stacklevel=2,
            )
            return torch.device("cpu")
        return torch.device("cuda")

    elif requested == "cpu":
        return torch.device("cpu")

    else:
        raise ValueError(
            f"Unknown device '{requested}'. Choose: auto | mps | cuda | cpu"
        )


def _mps_available() -> bool:
    """True if Apple Silicon MPS backend is available and built."""
    return (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )


def device_info(device: Optional[torch.device] = None) -> dict:
    """
    Return a dict of device information for logging.

    Example output (MPS):
        {
          "device": "mps",
          "platform": "Apple Silicon",
          "torch_version": "2.2.0",
          "mps_available": True,
          "cuda_available": False,
          "dtype": "float32"
        }
    """
    if device is None:
        device = get_device()

    info = {
        "device":         str(device),
        "torch_version":  torch.__version__,
        "mps_available":  _mps_available(),
        "cuda_available": torch.cuda.is_available(),
        "dtype":          "float32",  # MPS requires float32
    }

    if device.type == "mps":
        info["platform"] = "Apple Silicon (MPS)"
        info["note"]     = "Set PYTORCH_ENABLE_MPS_FALLBACK=1 for unsupported ops"
    elif device.type == "cuda":
        info["platform"] = f"CUDA — {torch.cuda.get_device_name(0)}"
        info["vram_gb"]  = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    else:
        info["platform"] = "CPU"

    return info


def print_device_info(device: Optional[torch.device] = None):
    """Pretty-print device info to stdout."""
    if device is None:
        device = get_device()
    info = device_info(device)
    icon = {"mps": "🍎", "cuda": "🖥️", "cpu": "💻"}.get(str(device).split(":")[0], "❓")
    print(f"\n{icon}  Device : {info['device']}  |  Platform : {info['platform']}")
    print(f"   PyTorch : {info['torch_version']}  |  dtype : {info['dtype']}")
    if "note" in info:
        print(f"   Note    : {info['note']}")
    if "vram_gb" in info:
        print(f"   VRAM    : {info['vram_gb']} GB")
    print()


# ─── Tensor helpers ───────────────────────────────────────────────────────────

def to_device(
    obj: Union[torch.Tensor, nn.Module, dict, list, tuple],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Union[torch.Tensor, nn.Module, dict, list, tuple]:
    """
    Recursively move tensors / modules to device, casting to float32.
    MPS does not support float64 — always use float32.

    Works on: Tensor, nn.Module, dict, list, tuple (recursive).
    """
    if isinstance(obj, torch.Tensor):
        if obj.is_floating_point():
            return obj.to(device=device, dtype=dtype)
        return obj.to(device=device)  # int tensors: no dtype cast

    elif isinstance(obj, nn.Module):
        return obj.to(device=device)

    elif isinstance(obj, dict):
        return {k: to_device(v, device, dtype) for k, v in obj.items()}

    elif isinstance(obj, (list, tuple)):
        moved = [to_device(v, device, dtype) for v in obj]
        return type(obj)(moved)

    return obj  # non-tensor scalars, strings etc.


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Safely convert a tensor to numpy, regardless of device (MPS/CUDA/CPU).
    MPS tensors must be moved to CPU before .numpy() call.
    """
    if tensor.device.type in ("mps", "cuda"):
        tensor = tensor.cpu()
    return tensor.detach().numpy()



# ─── FLOPs profiling ─────────────────────────────────────────────────────────

def profile_flops(model: nn.Module, input_shape: tuple) -> dict:
    """
    Profile FLOPs and parameter count using thop.
    Always runs on CPU (thop is not MPS/CUDA compatible).

    Args:
        model: nn.Module
        input_shape: tuple, e.g. (1, 1, 256) for a single 256-sample window

    Returns:
        dict with 'macs', 'params', 'flops_g'

    Example:
        >>> info = profile_flops(model, (1, 1, 2048))
        >>> print(f"FLOPs: {info['flops_g']:.3f} G  |  Params: {info['params']:,}")
    """
    try:
        from thop import profile, clever_format
    except ImportError:
        return {"error": "thop not installed. Run: pip install thop"}

    model_cpu = model.cpu()
    dummy_input = torch.randn(*input_shape).float()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        macs, params = profile(model_cpu, inputs=(dummy_input,), verbose=False)

    flops_g = macs * 2 / 1e9  # MACs × 2 ≈ FLOPs

    return {
        "macs":    int(macs),
        "params":  int(params),
        "flops_g": round(flops_g, 4),
        "params_m": round(params / 1e6, 3),
    }


def reset_cuda_peak_memory(device: Optional[torch.device] = None) -> None:
    """Reset CUDA peak memory counters (no-op on non-CUDA devices)."""
    if device is None:
        device = get_device()
    if device.type != "cuda":
        return
    idx = device.index if device.index is not None else torch.cuda.current_device()
    torch.cuda.reset_peak_memory_stats(idx)
    torch.cuda.synchronize(idx)


def cuda_peak_memory_mb(device: Optional[torch.device] = None) -> float | None:
    """Peak CUDA allocated memory in MB since last reset; ``None`` if not CUDA."""
    if device is None:
        device = get_device()
    if device.type != "cuda":
        return None
    idx = device.index if device.index is not None else torch.cuda.current_device()
    torch.cuda.synchronize(idx)
    return round(torch.cuda.max_memory_allocated(idx) / (1024 * 1024), 2)


def process_rss_mb() -> float:
    """Current process RSS in MB (NaN if psutil unavailable)."""
    try:
        import psutil

        return round(psutil.Process().memory_info().rss / (1024 * 1024), 2)
    except ImportError:
        return float("nan")


def measure_peak_rss_mb(fn, *args, **kwargs) -> tuple[object, float]:
    """
    Run ``fn(*args, **kwargs)`` and return ``(result, peak_rss_mb)``.

    Uses psutil when available; otherwise returns NaN for peak RSS.
    """
    peak_mb = float("nan")
    try:
        import psutil

        proc = psutil.Process()
        before = proc.memory_info().rss

        def _run():
            return fn(*args, **kwargs)

        result = _run()
        peak_mb = round(max(proc.memory_info().rss, before) / (1024 * 1024), 3)
        return result, peak_mb
    except ImportError:
        return fn(*args, **kwargs), peak_mb


# ─── Latency benchmarking ─────────────────────────────────────────────────────

def benchmark_latency(
    model: nn.Module,
    input_shape: tuple,
    device: Optional[torch.device] = None,
    n_runs: int = 1000,
    warmup: int = 50,
) -> dict:
    """
    Measure per-sample inference latency in milliseconds.
    Handles MPS, CUDA, and CPU correctly (MPS requires synchronization).

    Args:
        model:        nn.Module in eval mode
        input_shape:  tuple, e.g. (1, 1, 2048)
        device:       target device (auto-detected if None)
        n_runs:       number of timed inference calls
        warmup:       number of warm-up runs before timing

    Returns:
        dict with 'mean_ms', 'std_ms', 'min_ms', 'max_ms', 'device'

    Example:
        >>> stats = benchmark_latency(model, (1, 1, 2048))
        >>> print(f"Latency: {stats['mean_ms']:.2f} ± {stats['std_ms']:.2f} ms")
        >>> print(f"Target <70ms: {'✅' if stats['mean_ms'] < 70 else '❌'}")
    """
    import time

    if device is None:
        device = get_device()

    model = model.eval().to(device)
    dummy = torch.randn(*input_shape).float().to(device)

    # Warm-up
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
            _sync(device)

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(dummy)
            _sync(device)
            times.append((time.perf_counter() - t0) * 1000)

    import statistics
    return {
        "device":   str(device),
        "mean_ms":  round(statistics.mean(times), 3),
        "std_ms":   round(statistics.stdev(times), 3),
        "min_ms":   round(min(times), 3),
        "max_ms":   round(max(times), 3),
        "n_runs":   n_runs,
        "target_ok": statistics.mean(times) < 70.0,
    }


def _sync(device: torch.device):
    """Synchronize device — required for accurate timing on MPS and CUDA."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        # torch.mps.synchronize() available in PyTorch >= 2.0
        if hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()


# ─── Training helpers ─────────────────────────────────────────────────────────

def set_seed(seed: int = 42, device: Optional[torch.device] = None):
    """
    Set all random seeds for reproducibility across numpy, torch, and MPS/CUDA.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if device is None:
        device = get_device()

    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif device.type == "mps":
        # MPS does not have its own manual_seed; torch.manual_seed covers it
        pass


def get_scaler(device: Optional[torch.device] = None):
    """
    Return a GradScaler for mixed-precision training.
    CUDA: real AMP scaler.
    MPS: dummy scaler (MPS handles precision natively; no explicit AMP needed).
    CPU: dummy scaler.

    Usage:
        scaler = get_scaler(device)
        with torch.autocast(device_type=device.type, enabled=(device.type=="cuda")):
            loss = criterion(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    """
    if device is None:
        device = get_device()

    if device.type == "cuda":
        return torch.amp.GradScaler("cuda")
    else:
        # Dummy scaler that is a no-op — same API, safe to call on MPS/CPU
        return _DummyScaler()


class _DummyScaler:
    """No-op GradScaler for MPS and CPU (matches torch.cuda.amp.GradScaler API)."""
    def scale(self, loss):       return loss
    def step(self, optimizer):   optimizer.step()
    def update(self):            pass
    def unscale_(self, opt):     pass


# ─── DeviceAwareModule base class ────────────────────────────────────────────

class DeviceAwareModule(nn.Module):
    """
    Base class for MSPF-Net modules.
    Adds:
      - self.device property (current device of first parameter)
      - self.to_device(x) helper
      - MPS-safe forward type enforcement (float32)
    """

    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def to_device(self, x):
        return to_device(x, self.device)

    def safe_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Cast input to float32 before forward pass (required for MPS)."""
        return x.float()


# ─── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  MSPF-Net Device Resolver — Self Test")
    print("=" * 60)

    device = get_device("auto")
    print_device_info(device)

    # Test tensor movement
    x = torch.randn(4, 256).double()   # float64 — should be cast to float32
    x = to_device(x, device)
    print(f"  Tensor dtype after to_device : {x.dtype}  (expected: float32)")
    print(f"  Tensor device                : {x.device}")

    # Test to_numpy on MPS/CUDA/CPU
    arr = to_numpy(x)
    print(f"  to_numpy shape               : {arr.shape}")

    # Test seed setting
    set_seed(42, device)
    print(f"  Seed set for device          : {device}")

    # Test scaler API
    scaler = get_scaler(device)
    print(f"  Scaler type                  : {type(scaler).__name__}")

    print("\n  ✅ All device checks passed\n")
