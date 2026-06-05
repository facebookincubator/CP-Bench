# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Parameter randomization for cross-GPU SDC detection.

Randomizes test parameters across runs to maximize coverage when each
individual run is time-limited. Over many runs, this ensures all
hardware datapaths get exercised even if a single run can only test one.

Currently supports:
  --randomize-precision: randomly select one precision mode per run
"""

from __future__ import annotations

import logging
import random

from .precision import VALID_PRECISION_MODES

logger: logging.Logger = logging.getLogger("pytorch_op")

DEFAULT_PRECISION_POOL = ["fp32", "tf32", "fp16", "bf16"]


def randomize_precision(
    current_modes: list[str],
    pool: list[str] | None = None,
) -> list[str]:
    """Pick one random precision mode for this run.

    Args:
        current_modes: the precision modes currently configured (from CLI).
            Ignored when randomization is active — the random pick overrides.
        pool: precision modes to choose from. Defaults to DEFAULT_PRECISION_POOL.

    Returns:
        A single-element list with the randomly chosen precision mode.
    """
    pool = pool or DEFAULT_PRECISION_POOL
    pool = [m for m in pool if m in VALID_PRECISION_MODES]
    if not pool:
        logger.warning("No valid precision modes in pool, falling back to fp32")
        return ["fp32"]

    choice = random.choice(pool)
    logger.info(f"Randomized precision mode: {choice} (from pool: {pool})")
    return [choice]
