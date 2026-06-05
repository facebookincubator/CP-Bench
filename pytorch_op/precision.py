# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Precision mode configuration and deterministic execution setup.

Precision modes control which hardware datapath the GPU matmul unit uses.
Each exercises different silicon -- a GPU can have SDC in one mode but not
another (e.g., TF32/FP16/BF16 use reduced-precision datapaths while FP32
uses the full-precision datapath within the same matrix core).
"""

import random

import torch


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Supported precision modes and their hardware paths:
#   fp32  -- FP32 datapath in matrix core, full 32-bit mantissa
#   tf32  -- TF32 datapath in matrix core, truncated 10-bit mantissa
#   fp16  -- FP16 datapath in matrix core, half precision
#   bf16  -- BF16 datapath in matrix core, bfloat16
#   fp8   -- FP8 datapath in matrix core, 8-bit (E4M3/E5M2 variants)
VALID_PRECISION_MODES = {"fp32", "tf32", "fp16", "bf16", "fp8"}

# Map precision mode to torch dtype (for modes that change dtype).
PRECISION_MODE_DTYPE = {
    "fp32": torch.float32,
    "tf32": torch.float32,  # TF32 is a backend mode, not a separate dtype
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    # fp8 handled specially -- torch.float8_e4m3fn
}


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def set_precision_mode(mode: str) -> None:
    """Configure PyTorch backend for a specific precision mode.

    This controls which hardware datapath the matmul unit uses.
    Each mode exercises different silicon -- a GPU can have SDC
    in one mode but not another.
    """
    if mode == "fp32":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    elif mode == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif mode in ("fp16", "bf16", "fp8"):
        # Dtype-based -- tf32 backend setting is irrelevant
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        raise ValueError(
            f"Unknown precision mode: {mode}. Valid: {VALID_PRECISION_MODES}"
        )


def resolve_dtype(precision_mode: str) -> torch.dtype:
    """Return the torch dtype for a given precision mode."""
    if precision_mode == "fp8":
        return torch.float8_e4m3fn
    return PRECISION_MODE_DTYPE.get(precision_mode, torch.float32)


def enable_deterministic_mode() -> None:
    """Enable deterministic settings for reproducible results.

    Sets seeds and forces deterministic algorithms. Does NOT configure
    precision -- use set_precision_mode() for that.
    """
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
