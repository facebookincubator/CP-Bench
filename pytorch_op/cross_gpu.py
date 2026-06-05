# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Cross-GPU SDC detection: runs the same ops on all GPUs and compares results.

This catches deterministic-but-wrong errors that self-comparison misses.
Model-level cpbench detects SDC by comparing checksums across GPUs;
this module does the same at the operator level.

Always tests all GPUs on the box.
"""

import csv
import hashlib
import io
import logging
import signal
import subprocess
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Set, Tuple

import torch

from .ops import DTYPE_MAP, OP_REGISTRY, OpDef, PIPELINE_REGISTRY
from .precision import enable_deterministic_mode, resolve_dtype, set_precision_mode

logger = logging.getLogger("pytorch_op_crossgpu")


class _TimeLimitReached(Exception):
    """Raised by SIGALRM when the duration limit is hit."""


DEFAULT_SIZES = [
    (1024, 1024),
    (4096, 4096),
    (8192, 8192),
]


class CrossGPUDetector:
    """Runs ops on multiple GPUs and compares results across GPUs."""

    def __init__(
        self,
        gpu_ids: List[int],
        test_sizes=None,
        hash_inputs: bool = False,
    ):
        self.gpu_ids = gpu_ids
        self._test_sizes = test_sizes or DEFAULT_SIZES
        self.current_dtype = torch.float32
        self._failures: List[str] = []
        self._hash_inputs = hash_inputs
        self._gpu_disagreements: Dict[int, Set[int]] = {g: set() for g in gpu_ids}
        self.device: str = f"cuda:{gpu_ids[0]}"
        self._gpu_serials: Dict[int, str] = self._get_gpu_serials()
        self._run_max_diff: float = 0.0
        self._run_total_mismatched: int = 0
        self._run_total_elements: int = 0
        self._all_peers_disagree: bool = False

    @staticmethod
    def _get_gpu_serials() -> Dict[int, str]:
        """Get GPU serial numbers mapped to PyTorch rank.

        On AMD, PyTorch assigns GPU ranks by Node ID order (not DRM card
        index).  GPU rank 0 is the card with the lowest Node ID, rank 1
        the next, and so on.

        On NVIDIA, GPU index from nvidia-smi matches PyTorch rank
        directly (PCI bus order).
        """
        serials = CrossGPUDetector._get_amd_serials()
        if serials:
            return serials
        return CrossGPUDetector._get_nvidia_serials()

    @staticmethod
    def _get_amd_serials() -> Dict[int, str]:
        """Parse rocm-smi CSV, sort by Node ID, map sorted index to rank."""
        try:
            result = subprocess.run(
                ["rocm-smi", "--showproductname", "--showserial", "--showbus", "--csv"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0 or "Node ID" not in result.stdout:
                return {}
            reader = csv.DictReader(io.StringIO(result.stdout))
            entries = []
            for row in reader:
                node_id = row.get("Node ID", "").strip()
                serial = row.get("Serial Number", "").strip()
                if node_id and serial:
                    entries.append((int(node_id), serial))
            if not entries:
                return {}
            entries.sort(key=lambda x: x[0])
            return {rank: serial for rank, (_nid, serial) in enumerate(entries)}
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return {}

    @staticmethod
    def _get_nvidia_serials() -> Dict[int, str]:
        """Parse nvidia-smi CSV — GPU index already matches PyTorch rank."""
        serials: Dict[int, str] = {}
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,serial",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().splitlines():
                    parts = line.split(",")
                    if len(parts) == 2:
                        serials[int(parts[0].strip())] = parts[1].strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return serials

    def create_test_tensor(self, shape, dtype=None):
        dtype = dtype or self.current_dtype
        torch.manual_seed(42)
        if dtype in (torch.float16, torch.bfloat16):
            return torch.randn(shape, dtype=torch.float32, device=self.device).to(dtype)
        return torch.randn(shape, dtype=dtype, device=self.device)

    def _hash_tensor(self, t: torch.Tensor) -> str:
        """Compute a stable hash of a tensor's contents."""
        data = t.cpu().contiguous().float().numpy().tobytes()
        return hashlib.sha256(data).hexdigest()[:16]

    def _run_op_on_gpu(
        self, gpu_id: int, op_def: OpDef, size
    ) -> Optional[Tuple[torch.Tensor, Optional[str], float]]:
        """Run a single op on a specific GPU and return (result_on_cpu, input_hash, elapsed_ms)."""
        self.device = f"cuda:{gpu_id}"
        try:
            enable_deterministic_mode()
            # Capture input tensor for hashing
            torch.manual_seed(42)
            input_tensor = self.create_test_tensor(size)
            input_hash = self._hash_tensor(input_tensor) if self._hash_inputs else None
            del input_tensor

            # Re-seed and run the actual op with GPU timing
            operation = op_def.factory(self, size)
            with torch.cuda.device(gpu_id):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                result = operation()
                end_event.record()
                torch.cuda.synchronize()
                elapsed_ms = start_event.elapsed_time(end_event)

                if isinstance(result, torch.Tensor):
                    return (result.cpu(), input_hash, elapsed_ms)
                return (result, input_hash, elapsed_ms)
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.warning(f"  GPU {gpu_id}: OOM - skipping")
                torch.cuda.empty_cache()
                return None
            raise
        finally:
            torch.cuda.empty_cache()

    @staticmethod
    def _nan_aware_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
        """Compare two tensors treating NaN==NaN as True.

        IEEE 754 says NaN != NaN, so ``torch.equal`` returns False when
        both tensors contain NaN in the same positions.  This helper
        masks NaN positions before comparing the non-NaN elements.
        """
        nan_a = torch.isnan(a)
        nan_b = torch.isnan(b)
        if nan_a.any() or nan_b.any():
            # NaN positions must match exactly
            if not torch.equal(nan_a, nan_b):
                return False
            # Compare only non-NaN elements
            non_nan = ~nan_a
            if non_nan.any():
                return torch.equal(a[non_nan], b[non_nan])
            # Both are entirely NaN — treat as equal
            return True
        return torch.equal(a, b)

    def _report_mismatch(
        self,
        ref_gpu: int,
        other_gpu: int,
        op_name: str,
        ref: torch.Tensor,
        output: torch.Tensor,
        verbose: bool,
    ) -> None:
        """Log a cross-GPU mismatch and update run-level diff stats."""
        ref_f32 = ref.float()
        out_f32 = output.float()
        diff = torch.abs(ref_f32 - out_f32)
        # Mask out NaN positions so they don't poison diff stats
        nan_mask = torch.isnan(diff)
        if nan_mask.any():
            diff = diff.clone()
            diff[nan_mask] = 0.0
        mismatch_mask = diff > 0
        mismatches = int(mismatch_mask.sum().item())
        total = ref.numel()
        max_diff = diff.max().item()

        self._run_max_diff = max(self._run_max_diff, max_diff)
        self._run_total_mismatched += mismatches
        self._run_total_elements += total

        # Report NaN divergence separately when NaN positions differ
        nan_ref = torch.isnan(ref_f32)
        nan_out = torch.isnan(out_f32)
        nan_only_ref = int((nan_ref & ~nan_out).sum().item())
        nan_only_out = int((nan_out & ~nan_ref).sum().item())
        nan_note = ""
        if nan_only_ref or nan_only_out:
            nan_note = (
                f", NaN divergence: {nan_only_ref} in ref / {nan_only_out} in other"
            )

        logger.error(
            f"FAIL: GPU {ref_gpu} vs GPU {other_gpu}, {op_name}, "
            f"max_diff={max_diff:.2e}, "
            f"{mismatches}/{total} mismatched{nan_note}"
        )

        if verbose:
            self._log_verbose_diff(
                ref_gpu, other_gpu, ref, ref_f32, out_f32, diff, mismatch_mask
            )

        self._failures.append(
            f"{op_name} (GPU {ref_gpu} vs {other_gpu}: "
            f"max_diff={max_diff:.2e}, {mismatches}/{total} mismatched)"
        )

    def _compare_cross_gpu(
        self,
        op_name: str,
        results: Dict[int, Tuple[torch.Tensor, Optional[str]]],
        verbose: bool = False,
    ) -> bool:
        """Compare results across GPUs. Returns True if all match.

        Concise mode (default): one line per failure.
        Verbose mode: full element-level diffs, byte offsets, sample values.
        Also tracks per-GPU disagreements (all-pairs) for the summary.
        """
        gpu_ids = sorted(results.keys())
        if len(gpu_ids) < 2:
            logger.warning(
                f"  {op_name}: only {len(gpu_ids)} GPU(s) produced results, need >= 2"
            )
            return True

        ref_gpu = gpu_ids[0]
        ref, ref_input_hash = results[ref_gpu]
        all_match = True

        # Check input hashes
        if self._hash_inputs:
            for gpu_id in gpu_ids[1:]:
                _, other_hash = results[gpu_id]
                if other_hash != ref_input_hash:
                    logger.error(
                        f"FAIL: GPU {ref_gpu} vs GPU {gpu_id}, {op_name}, "
                        f"INPUT HASH MISMATCH -- memory corruption"
                    )
                    if verbose:
                        logger.error(
                            f"  GPU {ref_gpu}={ref_input_hash} vs "
                            f"GPU {gpu_id}={other_hash}"
                        )
                    self._failures.append(
                        f"{op_name} (GPU {ref_gpu} vs {gpu_id}: INPUT HASH MISMATCH)"
                    )
                    all_match = False

        # Compare outputs
        for gpu_id in gpu_ids[1:]:
            output, _ = results[gpu_id]
            if output.shape != ref.shape:
                logger.error(
                    f"FAIL: GPU {ref_gpu} vs GPU {gpu_id}, {op_name}, "
                    f"shape mismatch {ref.shape} vs {output.shape}"
                )
                self._failures.append(f"{op_name} (GPU {ref_gpu} vs {gpu_id}: shape)")
                all_match = False
                continue

            if not self._nan_aware_equal(ref, output):
                self._report_mismatch(ref_gpu, gpu_id, op_name, ref, output, verbose)
                all_match = False

        # Track per-GPU disagreements via all-pairs comparison
        if not all_match:
            self._track_disagreements(gpu_ids, results)

        return all_match

    def _log_verbose_diff(
        self,
        ref_gpu: int,
        other_gpu: int,
        ref: torch.Tensor,
        ref_f32: torch.Tensor,
        out_f32: torch.Tensor,
        diff: torch.Tensor,
        mismatch_mask: torch.Tensor,
    ) -> None:
        """Log detailed corruption analysis for verbose mode."""
        mismatches = int(mismatch_mask.sum().item())

        # Stuck-at-zero pattern
        zero_mask = (out_f32 == 0) & (ref_f32 != 0)
        zeroed_count = int(zero_mask.sum().item())
        if zeroed_count > 0:
            logger.error(
                f"  ZEROED VALUES: {zeroed_count}/{mismatches} corrupted "
                f"elements are zero (stuck-at-zero pattern)"
            )

        # Corruption geometry (capped to avoid log flood)
        max_lines = 10
        if mismatches > 0 and ref.dim() >= 2:
            mismatch_indices = torch.nonzero(mismatch_mask, as_tuple=False)
            if len(mismatch_indices) > 0 and ref.dim() == 2:
                rows = mismatch_indices[:, 0]
                cols = mismatch_indices[:, 1]
                unique_rows = rows.unique()
                lines = 0
                for row in unique_rows:
                    if lines >= max_lines:
                        logger.error(
                            f"  GEOMETRY: ... {len(unique_rows) - lines} more "
                            f"rows omitted"
                        )
                        break
                    row_cols = cols[rows == row].sort()[0]
                    start_col = row_cols[0].item()
                    byte_offset = start_col * ref.element_size()
                    if len(row_cols) > 1:
                        is_contiguous = (row_cols[1:] - row_cols[:-1] == 1).all().item()
                        aligned_32b = (byte_offset % 32) == 0
                        logger.error(
                            f"  GEOMETRY: row {row.item()}, "
                            f"cols {row_cols[0].item()}-{row_cols[-1].item()} "
                            f"({len(row_cols)} elements, "
                            f"{'contiguous' if is_contiguous else 'non-contiguous'}, "
                            f"byte_offset={byte_offset}, "
                            f"{'32B-aligned' if aligned_32b else 'NOT 32B-aligned'})"
                        )
                    else:
                        logger.error(
                            f"  GEOMETRY: row {row.item()}, col {start_col} "
                            f"(1 element, byte_offset={byte_offset})"
                        )
                    lines += 1

                # Flat byte offsets
                flat_indices = torch.nonzero(
                    mismatch_mask.flatten(), as_tuple=False
                ).squeeze()
                if flat_indices.dim() == 0:
                    flat_indices = flat_indices.unsqueeze(0)
                byte_offsets = flat_indices * ref.element_size()
                logger.error(
                    f"  BYTE OFFSETS of corrupted elements: "
                    f"{byte_offsets[:10].tolist()}"
                    f"{'...' if len(byte_offsets) > 10 else ''}"
                )

        # Top mismatches
        top_diffs = torch.topk(diff.flatten(), min(5, diff.numel()))
        for idx in top_diffs.indices:
            pos = idx.item()
            logger.error(
                f"  [{pos}]: "
                f"GPU{ref_gpu}={ref_f32.flatten()[pos].item():.10e}, "
                f"GPU{other_gpu}={out_f32.flatten()[pos].item():.10e}, "
                f"diff={diff.flatten()[pos].item():.10e}"
            )

    def _track_disagreements(
        self,
        gpu_ids: List[int],
        results: Dict[int, Tuple[torch.Tensor, Optional[str]]],
    ) -> None:
        """Track all-pairs GPU disagreements for the summary."""
        outputs = {g: results[g][0] for g in gpu_ids}
        for i, ga in enumerate(gpu_ids):
            for gb in gpu_ids[i + 1 :]:
                differs = outputs[ga].shape != outputs[
                    gb
                ].shape or not self._nan_aware_equal(outputs[ga], outputs[gb])
                if differs:
                    self._gpu_disagreements[ga].add(gb)
                    self._gpu_disagreements[gb].add(ga)
        if self._hash_inputs:
            hashes = {g: results[g][1] for g in gpu_ids}
            for i, ga in enumerate(gpu_ids):
                for gb in gpu_ids[i + 1 :]:
                    if hashes[ga] is not None and hashes[ga] != hashes[gb]:
                        self._gpu_disagreements[ga].add(gb)
                        self._gpu_disagreements[gb].add(ga)

    def run(
        self,
        categories: Optional[List[str]] = None,
        dtypes: Optional[List[str]] = None,
        iterations: int = 1,
        run_pipelines: bool = False,
        fail_fast: bool = False,
        verbose: bool = False,
        precision_modes: Optional[List[str]] = None,
        duration_limit: Optional[int] = None,
    ) -> dict:
        """Run cross-GPU comparison tests.

        For each op, runs it on every GPU and compares results.
        Repeats ``iterations`` times to catch intermittent SDC.

        Returns a dict with pass/fail counts, timing, and per-GPU
        disagreement info for identifying bad GPUs.
        """
        dtypes = dtypes or ["float32"]
        resolved_dtypes = [DTYPE_MAP[d.strip()] for d in dtypes]
        precision_modes = precision_modes or ["fp32"]

        logger.info(f"Cross-GPU SDC detection on GPUs: {self.gpu_ids}")
        logger.info(f"Iterations per op: {iterations}")
        logger.info(f"Dtypes: {dtypes}")
        logger.info(f"Precision modes: {precision_modes}")
        logger.info(f"Sizes: {self._test_sizes}")
        if fail_fast:
            logger.info("Fail-fast: ON")
        if verbose:
            logger.info("Verbose: ON")
        if duration_limit is not None:
            logger.info(f"Duration limit: {duration_limit}s (hard cap)")

        start_time = time.time()
        self._failures = []
        self._gpu_disagreements = {g: set() for g in self.gpu_ids}
        self._run_max_diff = 0.0
        self._run_total_mismatched = 0
        self._run_total_elements = 0
        self._all_peers_disagree = False
        total = 0
        passed = 0
        stopped = False
        time_limited = False

        # Set up SIGALRM for hard duration cap — interrupts immediately,
        # even mid-CUDA-call, no waiting for any op/kernel to finish.
        old_handler = None
        if duration_limit is not None:

            def _alarm_handler(signum, frame):
                raise _TimeLimitReached()

            old_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(duration_limit)

        try:
            # --- Individual op tests ---
            for pmode in precision_modes:
                if stopped:
                    break
                set_precision_mode(pmode)
                mode_dtype = resolve_dtype(pmode)
                pmode_label = pmode.upper()

                if len(precision_modes) > 1:
                    logger.info(f"=== Precision mode: {pmode_label} ===")

                for dtype in resolved_dtypes:
                    if stopped:
                        break
                    # Dtype-based precision modes fix the dtype
                    if pmode in ("fp16", "bf16", "fp8"):
                        self.current_dtype = mode_dtype
                    else:
                        self.current_dtype = dtype
                    dtype_name = str(self.current_dtype).replace("torch.", "")
                    if len(resolved_dtypes) > 1 and pmode in ("fp32", "tf32"):
                        logger.info(f"  === Dtype: {dtype_name} ===")

                    for size in self._test_sizes:
                        if stopped:
                            break
                        logger.info(f"--- Size: {size} ({pmode_label}) ---")

                        grouped: OrderedDict[str, List[OpDef]] = OrderedDict()
                        for op_def in OP_REGISTRY:
                            if categories and op_def.category not in categories:
                                continue
                            grouped.setdefault(op_def.category, []).append(op_def)

                        for category, ops in grouped.items():
                            if stopped:
                                break
                            logger.info(f"  [{category.replace('_', ' ').title()}]")
                            for op_def in ops:
                                if stopped:
                                    break
                                op_ok = True
                                for it in range(iterations):
                                    results = {}
                                    for gpu_id in self.gpu_ids:
                                        r = self._run_op_on_gpu(gpu_id, op_def, size)
                                        if r is not None:
                                            result_tensor, input_hash, elapsed_ms = r
                                            results[gpu_id] = (
                                                result_tensor,
                                                input_hash,
                                            )

                                    total += 1
                                    label = op_def.name
                                    if len(precision_modes) > 1:
                                        label = f"{label} [{pmode_label}]"
                                    if iterations > 1:
                                        label = f"{label} iter {it + 1}/{iterations}"
                                    if self._compare_cross_gpu(
                                        label, results, verbose=verbose
                                    ):
                                        passed += 1
                                    else:
                                        op_ok = False
                                        if fail_fast:
                                            stopped = True
                                            break

                                if op_ok:
                                    iter_note = (
                                        f" across {iterations} iterations"
                                        if iterations > 1
                                        else ""
                                    )
                                    logger.info(
                                        f"    PASSED {op_def.name} "
                                        f"(all {len(self.gpu_ids)} GPUs match"
                                        f"{iter_note})"
                                    )

                    # Dtype-based modes run once (dtype is fixed by mode)
                    if pmode in ("fp16", "bf16", "fp8"):
                        break

            # --- Pipeline tests ---
            if run_pipelines and PIPELINE_REGISTRY and not stopped:
                for pmode in precision_modes:
                    if stopped:
                        break
                    set_precision_mode(pmode)
                    mode_dtype = resolve_dtype(pmode)
                    self.current_dtype = (
                        mode_dtype
                        if pmode in ("fp16", "bf16", "fp8")
                        else resolved_dtypes[0]
                    )
                    pmode_label = pmode.upper()
                    logger.info(f"--- Pipeline Tests ({pmode_label}) ---")
                    pipeline_size = (1024, 1024)

                    for _pidx, pipeline_def in enumerate(PIPELINE_REGISTRY):
                        if stopped:
                            break
                        for it in range(iterations):
                            results = {}
                            for gpu_id in self.gpu_ids:
                                self.device = f"cuda:{gpu_id}"
                                try:
                                    enable_deterministic_mode()
                                    set_precision_mode(pmode)
                                    operation = pipeline_def.factory(
                                        self, pipeline_size
                                    )
                                    with torch.cuda.device(gpu_id):
                                        result = operation()
                                        if isinstance(result, torch.Tensor):
                                            results[gpu_id] = (result.cpu(), None)
                                except RuntimeError as e:
                                    if "out of memory" in str(e):
                                        logger.warning(
                                            f"  GPU {gpu_id}: OOM on pipeline - skipping"
                                        )
                                        torch.cuda.empty_cache()
                                    else:
                                        raise
                                finally:
                                    torch.cuda.empty_cache()

                            total += 1
                            plabel = f"[Pipeline] {pipeline_def.name}"
                            if len(precision_modes) > 1:
                                plabel = f"{plabel} [{pmode_label}]"
                            if iterations > 1:
                                plabel = f"{plabel} iter {it + 1}/{iterations}"
                            if self._compare_cross_gpu(
                                plabel,
                                results,
                                verbose=verbose,
                            ):
                                passed += 1
                            else:
                                if fail_fast:
                                    stopped = True
                                    break

                        if not stopped:
                            check_label = f"[Pipeline] {pipeline_def.name}"
                            if len(precision_modes) > 1:
                                check_label = f"{check_label} [{pmode_label}]"
                            pipeline_failed = any(
                                check_label in f for f in self._failures
                            )
                            if not pipeline_failed:
                                mode_note = (
                                    f" [{pmode_label}]"
                                    if len(precision_modes) > 1
                                    else ""
                                )
                                logger.info(
                                    f"    PASSED [Pipeline] {pipeline_def.name} "
                                    f"(all GPUs match{mode_note})"
                                )

        except _TimeLimitReached:
            time_limited = True
            logger.info(f"\nDuration limit ({duration_limit}s) reached — hard stop")
        finally:
            if duration_limit is not None:
                signal.alarm(0)
                if old_handler is not None:
                    signal.signal(signal.SIGALRM, old_handler)

        elapsed = time.time() - start_time
        # Use len(self._failures) rather than total-passed: if SIGALRM fires
        # after total+=1 but before _compare_cross_gpu completes, total-passed
        # would be 1 with an empty failures list (false positive FAIL result).
        failed = len(self._failures)
        all_peers_disagree = self._is_all_peers_disagree()
        self._log_summary(
            elapsed, total, passed, failed, stopped, time_limited, all_peers_disagree
        )
        self._all_peers_disagree = all_peers_disagree

        if all_peers_disagree:
            passed = total
            failed = 0
            self._failures.clear()

        return {
            "total_comparisons": total,
            "passed": passed,
            "failed": failed,
            "elapsed_seconds": round(elapsed, 2),
            "time_limited": time_limited,
            "max_diff": self._run_max_diff,
            "total_mismatched": self._run_total_mismatched,
            "total_elements": self._run_total_elements,
            "failures": list(self._failures),
            "all_peers_disagree": self._all_peers_disagree,
            "gpu_disagreements": {
                g: sorted(peers) for g, peers in self._gpu_disagreements.items()
            },
        }

    def _is_all_peers_disagree(self) -> bool:
        """Check if every GPU disagrees with every other GPU.

        When all GPUs show max disagreement (N-1 out of N-1 peers),
        it indicates a systemic issue (e.g., non-determinism, driver
        bug, library mismatch) rather than a single bad GPU.  Real SDC
        has one GPU disagreeing with all others while the rest agree
        among themselves.

        Requires N >= 3 GPUs. With only 2 GPUs (n_peers=1), both a
        single bad GPU and a systemic issue look identical — every GPU
        shows 1/1 peer disagreement — so we cannot distinguish them and
        conservatively report as SDC.
        """
        n_peers = len(self.gpu_ids) - 1
        if n_peers < 2:
            return False
        return all(
            len(self._gpu_disagreements.get(g, set())) == n_peers for g in self.gpu_ids
        )

    def _log_summary(
        self,
        elapsed: float,
        total: int,
        passed: int,
        failed: int,
        stopped: bool,
        time_limited: bool = False,
        all_peers_disagree: bool = False,
    ) -> None:
        """Print the final per-GPU summary."""
        if self._failures:
            n_peers = len(self.gpu_ids) - 1

            if all_peers_disagree:
                logger.warning(
                    "\nALL GPUs DISAGREE — systemic issue, not SDC. "
                    "Every GPU produced different results, which indicates "
                    "non-determinism, driver bug, or library mismatch rather "
                    "than a single faulty GPU."
                )
                logger.warning(
                    f"  {len(self._failures)} comparison(s) failed across "
                    f"{len(self.gpu_ids)} GPUs in {elapsed:.1f}s"
                )
                return

            logger.error("\nSDC Summary:")
            for gpu_id in sorted(self.gpu_ids):
                serial = self._gpu_serials.get(gpu_id, "unknown")
                serial_str = f" [S/N: {serial}]" if serial != "unknown" else ""
                n_disagree = len(self._gpu_disagreements.get(gpu_id, set()))
                if n_disagree > 0:
                    label = f"FAIL ({n_disagree}/{n_peers} peers disagree)"
                    if n_disagree == n_peers:
                        label += " <-- BAD GPU"
                    logger.error(f"  GPU {gpu_id}{serial_str}: {label}")
                else:
                    logger.info(
                        f"  GPU {gpu_id}{serial_str}: PASS (0/{n_peers} peers disagree)"
                    )

            if stopped and not time_limited:
                logger.error(
                    f"\nSDC DETECTED -- stopped after first failure "
                    f"({passed} passed before failure, {elapsed:.1f}s)"
                )
            elif time_limited:
                logger.error(
                    f"\nCROSS-GPU SDC DETECTED: {len(self._failures)} failures "
                    f"in {total} comparisons across {len(self.gpu_ids)} GPUs "
                    f"({elapsed:.1f}s) (time-limited)"
                )
            else:
                logger.error(
                    f"\nCROSS-GPU SDC DETECTED: {len(self._failures)} failures "
                    f"in {total} comparisons across {len(self.gpu_ids)} GPUs "
                    f"({elapsed:.1f}s)"
                )
        else:
            suffix = " (time-limited)" if time_limited else ""
            logger.info(
                f"\nALL CROSS-GPU TESTS PASSED - "
                f"{total} comparisons across {len(self.gpu_ids)} GPUs "
                f"in {elapsed:.1f}s{suffix}"
            )
