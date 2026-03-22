import os
import warnings
from collections.abc import Callable

import torch


_WARNED = False


def safe_compile(fn: Callable) -> Callable:
    global _WARNED

    if os.environ.get("MINI_VLLM_DISABLE_TORCH_COMPILE") == "1":
        return fn

    try:
        return torch.compile(fn)
    except Exception as exc:  # pragma: no cover - depends on local torch/triton environment
        if not _WARNED:
            warnings.warn(f"torch.compile unavailable, falling back to eager functions: {exc}", RuntimeWarning)
            _WARNED = True
        return fn
