# Copyright (c) Meta Platforms, Inc. and affiliates.

"""GPU SDC Detector for PyTorch operator-level testing.

Standalone class that runs registered ops multiple times with deterministic
settings and compares outputs for bit-exact reproducibility.
"""

import logging
import random
import time
from collections import OrderedDict
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

from .ops import DTYPE_MAP, OP_REGISTRY, OpDef, PIPELINE_REGISTRY

logger = logging.getLogger("pytorch_op_test")


class GPUSDCDetector:
    """Detects Silent Data Corruption by checking op determinism on GPU."""

    # Default tensor sizes: progressive memory coverage.
    # Use --sizes for custom shapes (e.g. rectangular: 1024x16384).
    DEFAULT_SIZES = [
        (1024, 1024),
        (4096, 4096),
        (8192, 8192),
    ]

    def __init__(
        self,
        gpu_id: int = 0,
        num_iterations: int = 3,
        test_sizes: Optional[List[Tuple[int, int]]] = None,
    ):
        self.gpu_id = gpu_id
        self.num_iterations = num_iterations
        self.current_dtype = torch.float32
        self.device = f"cuda:{gpu_id}"
        self._failed_operations: List[str] = []
        self._test_sizes = test_sizes or self.DEFAULT_SIZES

    def create_test_tensor(
        self, shape: Tuple[int, ...], dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """Create a test tensor on the GPU with fixed seed."""
        dtype = dtype or self.current_dtype
        torch.manual_seed(42)
        if dtype in (torch.float16, torch.bfloat16):
            return torch.randn(shape, dtype=torch.float32, device=self.device).to(dtype)
        return torch.randn(shape, dtype=dtype, device=self.device)

    def _resolve_dtypes(self, dtypes: Optional[List[str]]) -> List[torch.dtype]:
        """Validate and resolve dtype strings to torch.dtype objects."""
        if dtypes is None:
            dtypes = ["float32"]
        resolved = []
        for dt_name in dtypes:
            dt_name = dt_name.strip()
            if dt_name not in DTYPE_MAP:
                raise ValueError(
                    f"Unknown dtype: {dt_name}. "
                    f"Available: {', '.join(DTYPE_MAP.keys())}"
                )
            resolved.append(DTYPE_MAP[dt_name])
        return resolved

    def _check_gpu(self):
        """Validate that the target GPU is available."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        if self.gpu_id >= torch.cuda.device_count():
            raise RuntimeError(
                f"GPU {self.gpu_id} not found. Available: {torch.cuda.device_count()}"
            )

    def _run_op_tests(
        self, categories: Optional[List[str]], resolved_dtypes: List[torch.dtype]
    ) -> Tuple[int, int]:
        """Run all registered op tests across dtypes and sizes. Returns (total, passed)."""
        total_ops = 0
        passed_ops = 0
        for dtype in resolved_dtypes:
            self.current_dtype = dtype
            if len(resolved_dtypes) > 1:
                logger.info(f"Testing dtype: {dtype}")
            for size in self._test_sizes:
                logger.info(f"Testing with tensor size: {size}")
                for category, ops in self._get_ops_by_category(categories):
                    cat_label = category.replace("_", " ").title()
                    logger.info(f"--- {cat_label} Operations ---")
                    for op_def in ops:
                        operation = op_def.factory(self, size)
                        total_ops += 1
                        if self._test_operation(op_def.name, operation):
                            passed_ops += 1
        return total_ops, passed_ops

    def _run_pipeline_tests(self) -> Tuple[int, int]:
        """Run composite pipeline tests. Returns (total, passed)."""
        total_ops = 0
        passed_ops = 0
        logger.info("--- Composite Pipeline Tests ---")
        size = (1024, 1024)
        for pipeline_def in PIPELINE_REGISTRY:
            operation = pipeline_def.factory(self, size)
            total_ops += 1
            if self._test_operation(f"[Pipeline] {pipeline_def.name}", operation):
                passed_ops += 1
        return total_ops, passed_ops

    def _log_results(self, total_ops: int, elapsed: float):
        """Log final pass/fail summary."""
        if self._failed_operations:
            logger.error(
                f"FAILED: {len(self._failed_operations)} operation(s) showed corruption"
            )
            for i, op in enumerate(self._failed_operations, 1):
                logger.error(f"  {i}. {op}")
            logger.error(f"GPU {self.gpu_id} is producing non-deterministic results")
        else:
            logger.info(
                f"ALL TESTS PASSED - GPU {self.gpu_id} deterministic across "
                f"{total_ops} operations in {elapsed:.1f}s"
            )

    def run_all_tests(
        self,
        categories: Optional[List[str]] = None,
        dtypes: Optional[List[str]] = None,
        run_pipelines: bool = False,
    ) -> dict:
        """Run all registered op tests and return results dict.

        Returns:
            dict with keys: ops_total, ops_passed, ops_failed,
            elapsed_seconds, failed_operations
        """
        resolved_dtypes = self._resolve_dtypes(dtypes)
        self._check_gpu()
        self._enable_deterministic_mode()

        logger.info(
            f"Testing GPU {self.gpu_id}: {torch.cuda.get_device_name(self.gpu_id)}"
        )
        logger.info(f"Iterations per operation: {self.num_iterations}")

        start_time = time.time()
        self._failed_operations = []

        total_ops, passed_ops = self._run_op_tests(categories, resolved_dtypes)

        if run_pipelines and PIPELINE_REGISTRY:
            self.current_dtype = resolved_dtypes[0]
            pipe_total, pipe_passed = self._run_pipeline_tests()
            total_ops += pipe_total
            passed_ops += pipe_passed

        elapsed = time.time() - start_time
        self._log_results(total_ops, elapsed)

        return {
            "ops_total": total_ops,
            "ops_passed": passed_ops,
            "ops_failed": len(self._failed_operations),
            "elapsed_seconds": round(elapsed, 2),
            "failed_operations": list(self._failed_operations),
        }

    def _enable_deterministic_mode(self):
        """Enable deterministic settings for reproducible results."""
        seed = 42
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _run_operation_on_gpu(self, operation: Callable) -> torch.Tensor:
        """Run an operation on the GPU and return result on CPU."""
        with torch.cuda.device(self.gpu_id):
            result = operation()
            if isinstance(result, torch.Tensor):
                return result.cpu()
            return result

    def _test_operation(self, op_name: str, operation: Callable) -> bool:
        """Test an operation by running it multiple times and comparing."""
        try:
            outputs = []
            for _ in range(self.num_iterations):
                self._enable_deterministic_mode()
                output = self._run_operation_on_gpu(operation)
                outputs.append(output)

            return self._compare_outputs(outputs, op_name)
        except Exception as e:
            logger.error(f"EXCEPTION in {op_name}: {str(e)}")
            self._failed_operations.append(op_name)
            return False

    def _compare_outputs(self, outputs: List[torch.Tensor], op_name: str) -> bool:
        """Compare multiple outputs from the same operation."""
        reference = outputs[0]

        for _i, output in enumerate(outputs[1:], start=1):
            if output.shape != reference.shape:
                logger.error(f"SHAPE MISMATCH in {op_name}")
                self._failed_operations.append(op_name)
                return False

            # Bit-exact comparison — any difference at all is SDC.
            # Use NaN-aware comparison: IEEE 754 says NaN != NaN, so
            # torch.equal returns False when both tensors have NaN in
            # the same positions.  Treat matching NaN positions as equal.
            nan_ref = torch.isnan(reference)
            nan_out = torch.isnan(output)
            has_nan = nan_ref.any() or nan_out.any()
            if has_nan:
                nan_positions_match = torch.equal(nan_ref, nan_out)
                non_nan = ~nan_ref
                non_nan_equal = not non_nan.any() or torch.equal(
                    reference[non_nan], output[non_nan]
                )
                tensors_equal = nan_positions_match and non_nan_equal
            else:
                tensors_equal = torch.equal(reference, output)

            if not tensors_equal:
                # Upcast to float32 only for the detailed mismatch report
                ref_f32 = reference.float()
                out_f32 = output.float()
                diff = torch.abs(ref_f32 - out_f32)
                # Mask NaN positions so they don't poison stats
                nan_mask = torch.isnan(diff)
                if nan_mask.any():
                    diff = diff.clone()
                    diff[nan_mask] = 0.0
                mismatches = int((diff > 0).sum().item())
                total = reference.numel()
                max_diff = diff.max().item()
                logger.error(
                    f"CORRUPTION DETECTED in {op_name}: "
                    f"max_diff={max_diff:.2e}, "
                    f"mismatched={mismatches}/{total} "
                    f"({100 * mismatches / total:.2f}%)"
                )
                self._print_detailed_mismatch(ref_f32, out_f32, diff, op_name)
                self._failed_operations.append(op_name)
                return False

        logger.info(
            f"PASSED {op_name} (all {self.num_iterations} iterations identical)"
        )
        return True

    def _print_detailed_mismatch(
        self,
        reference: torch.Tensor,
        output: torch.Tensor,
        diff: torch.Tensor,
        op_name: str,
    ):
        """Print detailed information about the mismatch."""
        max_diff_idx = diff.argmax()
        max_diff_pos = np.unravel_index(max_diff_idx.item(), diff.shape)

        logger.error(f"  Location of max difference: {max_diff_pos}")
        logger.error(f"  Expected: {reference.flatten()[max_diff_idx].item():.10e}")
        logger.error(f"  Actual:   {output.flatten()[max_diff_idx].item():.10e}")

        num_samples = min(10, diff.numel())
        top_diffs = torch.topk(diff.flatten(), num_samples)
        logger.error(f"  Top {num_samples} mismatched values:")
        for idx in top_diffs.indices:
            pos = np.unravel_index(idx.item(), diff.shape)
            logger.error(
                f"    {pos}: "
                f"expected={reference.flatten()[idx].item():.10e}, "
                f"actual={output.flatten()[idx].item():.10e}, "
                f"diff={diff.flatten()[idx].item():.10e}"
            )

        logger.error(
            f"  Reference: mean={reference.mean().item():.6e}, "
            f"std={reference.std().item():.6e}"
        )
        logger.error(
            f"  Corrupted: mean={output.mean().item():.6e}, "
            f"std={output.std().item():.6e}"
        )

    def _get_ops_by_category(
        self, categories: Optional[List[str]] = None
    ) -> List[Tuple[str, List[OpDef]]]:
        """Return ops grouped by category, preserving registration order."""
        grouped: OrderedDict[str, List[OpDef]] = OrderedDict()
        for op_def in OP_REGISTRY:
            if categories and op_def.category not in categories:
                continue
            grouped.setdefault(op_def.category, []).append(op_def)
        return list(grouped.items())

    def list_ops(self):
        """Print all registered ops grouped by category."""
        grouped: OrderedDict[str, List[str]] = OrderedDict()
        for op_def in OP_REGISTRY:
            grouped.setdefault(op_def.category, []).append(op_def.name)

        print("Registered operations:\n")
        total = 0
        for category, names in grouped.items():
            cat_label = category.replace("_", " ").title()
            print(f"  {cat_label} ({len(names)}):")
            for name in names:
                print(f"    - {name}")
            total += len(names)
            print()

        print(f"Total: {total} operations across {len(grouped)} categories")

        if PIPELINE_REGISTRY:
            print(f"\nComposite pipelines ({len(PIPELINE_REGISTRY)}):")
            for p in PIPELINE_REGISTRY:
                print(f"    - {p.name}")

        print(f"\nAvailable categories: {', '.join(grouped.keys())}")
        print(f"Available dtypes: {', '.join(DTYPE_MAP.keys())}")
