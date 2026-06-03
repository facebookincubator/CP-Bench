# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Unified entry point for PyTorch operator-level SDC detection.

Modes:
  self       - Self-consistency on a single GPU (original behavior)
  cross-gpu  - Cross-GPU operator comparison (recommended for SDC detection)
  sweep      - Systematic HBM region coverage via staircase operator tests

Presets (for cross-gpu mode, override --iterations/--sizes/--categories):
  quick      - 1 iter, all ops, 4kx4k, hash-inputs                  (~1-2 min)
  standard   - 3 iters, all ops+pipelines, 4kx4k, hash-inputs      (~5 min)
  thorough   - 10 iters, all ops+pipelines, 4kx4k + sweep           (~25-30 min)

Memory warmup is automatic: each cross-gpu run randomizes warmup between
0% and a calculated max (leaving room for test tensors + CUDA overhead).
This varies which HBM regions are tested across runs. Sweep mode manages
memory deterministically via staircase allocation.

By default, runs all ops to completion and prints a per-GPU summary
identifying bad GPU(s). Use --fail-fast to stop on first failure,
--verbose/-v for full element-level detail.

Usage:
  python -m pytorch_op.main --preset quick               All GPUs, quick check
  python -m pytorch_op.main --preset quick --fail-fast    Stop on first failure
  python -m pytorch_op.main --preset quick -v             Full detail on failures
  python -m pytorch_op.main --mode sweep                  Systematic full-HBM scan
  python -m pytorch_op.main --mode self                   Self-test all GPUs
  python -m pytorch_op.main --mode cross-gpu --hash-inputs  Cross-GPU, all GPUs
  python -m pytorch_op.main --mode cross-gpu --precision-modes tf32  Test matrix cores
  python -m pytorch_op.main --preset standard --duration-limit 120   Hard stop after 2 min
  python -m pytorch_op.main --list-ops
"""

import argparse
import logging
import os
import random
import sys

logger = logging.getLogger("pytorch_op")

# Preset configurations for cross-gpu mode
PRESETS = {
    "quick": {
        "iterations": 1,
        "sizes": [(4096, 4096)],
        "categories": None,  # all
        "hash_inputs": True,
        "pipelines": False,
    },
    "standard": {
        "iterations": 3,
        "sizes": [(4096, 4096)],
        "categories": None,  # all
        "hash_inputs": True,
        "pipelines": True,
    },
    "thorough": {
        "iterations": 10,
        "sizes": [(4096, 4096)],
        "categories": None,  # all
        "hash_inputs": True,
        "pipelines": True,
        "sweep": True,
        "sweep_chunk_gb": 8.0,
    },
}


def _parse_sizes(sizes_str):
    """Parse comma-separated size specs like '4096x4096,14000x14000'."""
    sizes = []
    for s in sizes_str.split(","):
        s = s.strip()
        if "x" in s:
            parts = s.split("x")
            sizes.append((int(parts[0]), int(parts[1])))
        else:
            n = int(s)
            sizes.append((n, n))
    return sizes


def _get_gpu_ids():
    """Return all GPU IDs on the box."""
    import torch

    return list(range(torch.cuda.device_count()))


def _random_warmup(gpu_ids, test_sizes, element_size=4):
    """Randomize memory warmup to vary which HBM regions are tested.

    Each run allocates a random amount of GPU memory (0% to max safe %)
    before testing. This varies which HBM addresses test tensors land in,
    improving coverage across multiple runs.

    Max warmup is calculated dynamically:
      max_warmup = free_mem - (2 * largest_tensor) - cuda_overhead

    This leaves room for 2x the largest test tensor (input + output) plus
    buffer for CUDA context/workspace. Adapts to actual GPU memory and
    test config — no hardcoded percentages.

    Sweep mode does NOT use warmup — it manages memory deterministically
    via staircase allocation.
    """
    import torch

    # Largest tensor in bytes from test config
    max_elements = max(m * n for m, n in test_sizes)
    largest_tensor_bytes = max_elements * element_size
    # Reserve enough free memory for test ops to run without OOM.
    # Backward ops and pipelines allocate multiple large intermediates
    # (activations, gradients, optimizer state), so we need more than
    # just 2x the test tensor. Use at least 4 GB for op execution.
    op_buffer = max(2 * largest_tensor_bytes, 4 * (1024**3))
    # CUDA overhead: cuBLAS workspace, RNG state, caching allocator metadata
    cuda_overhead = 2 * (1024**3)  # 2 GB

    # Use min free memory across GPUs to determine safe max
    min_free = min(torch.cuda.mem_get_info(g)[0] for g in gpu_ids)
    max_warmup = max(0, min_free - op_buffer - cuda_overhead)

    if max_warmup == 0:
        logger.info("Warmup: not enough free memory, skipping")
        return {}

    # Same random amount for all GPUs (tests same HBM depth)
    warmup_bytes = random.randint(0, int(max_warmup))
    total_mem = torch.cuda.get_device_properties(gpu_ids[0]).total_memory
    warmup_pct = warmup_bytes / total_mem * 100
    max_pct = max_warmup / total_mem * 100

    logger.info(
        f"Warmup: {warmup_pct:.1f}% "
        f"({warmup_bytes / 1e9:.1f} / {total_mem / 1e9:.1f} GB, "
        f"max {max_pct:.1f}%)"
    )

    held_tensors = {}
    chunk_elements = 256 * 1024 * 1024  # 1GB chunks in float32
    for gpu_id in gpu_ids:
        device = f"cuda:{gpu_id}"
        tensors = []
        allocated = 0
        while allocated < warmup_bytes:
            remaining = warmup_bytes - allocated
            n_elements = min(chunk_elements, remaining // 4)
            if n_elements <= 0:
                break
            try:
                t = torch.randn(n_elements, device=device)
                tensors.append(t)
                allocated += n_elements * 4
            except torch.cuda.OutOfMemoryError:
                break
        held_tensors[gpu_id] = tensors
        logger.info(
            f"  GPU {gpu_id}: {allocated / 1e9:.1f} GB allocated, "
            f"{torch.cuda.mem_get_info(gpu_id)[0] / 1e9:.1f} GB free"
        )

    return held_tensors


def _sweep_cross_gpu(args, preset=None):
    """Run cross-GPU operator tests with systematic HBM coverage.

    Staircase approach — progressively fills GPU memory, testing each new
    region before filling it:
    1. Run operator tests with no fill → tensors land in lowest region
    2. Fill region 0 (hold chunk), run tests → tensors land in region 1
    3. Fill region 1 (hold chunk), run tests → tensors land in region 2
    4. Continue until GPU memory is exhausted

    Why staircase instead of sliding-window:
    Filling ALL memory at once forces CUDA internals (cuBLAS workspace,
    RNG state, caching allocator metadata) into corrupted HBM regions,
    which taints every subsequent operation regardless of where test tensors
    land. The staircase avoids this by never filling beyond the current
    test boundary — CUDA internals stay in the earliest (typically clean)
    regions.
    """
    import torch

    from .cross_gpu import CrossGPUDetector

    # Resolve config from preset
    iterations = args.iterations
    test_sizes = _parse_sizes(args.sizes) if args.sizes else None
    categories = None
    if args.categories:
        categories = [c.strip() for c in args.categories.split(",")]
    hash_inputs = args.hash_inputs
    run_pipelines = args.pipelines

    if preset:
        cfg = PRESETS[preset]
        if args.iterations == 1:
            iterations = cfg["iterations"]
        if test_sizes is None:
            test_sizes = cfg["sizes"]
        if categories is None:
            categories = cfg["categories"]
        hash_inputs = hash_inputs or cfg["hash_inputs"]
        run_pipelines = run_pipelines or cfg["pipelines"]

    gpu_ids = _get_gpu_ids()
    if len(gpu_ids) < 2:
        # Sweep requires cross-GPU comparison — skip gracefully on single-GPU
        # platforms (e.g. Grace Hopper GH200) to avoid false SDC reports.
        logger.info(
            f"{len(gpu_ids)} GPU(s) detected — skipping sweep mode "
            "(requires 2+ GPUs for cross-GPU comparison)"
        )
        return 0

    dtypes = [d.strip() for d in args.dtypes.split(",")]
    sweep_chunk_gb = args.sweep_chunk_gb
    chunk_elements = int(sweep_chunk_gb * 1e9 / 4)  # float32 = 4 bytes

    # Determine how many regions we can test
    free_mem = min(torch.cuda.mem_get_info(g)[0] for g in gpu_ids)
    n_regions = int(free_mem / (sweep_chunk_gb * 1e9))
    if n_regions == 0:
        print("ERROR: not enough free GPU memory for sweep", file=sys.stderr)
        return 1

    logger.info(
        f"Sweep: staircase through ~{n_regions} x {sweep_chunk_gb} GB regions "
        f"({n_regions * sweep_chunk_gb:.0f} GB per GPU)"
    )
    logger.info(f"Sweep: testing {n_regions} memory regions...\n")

    # Staircase: progressively fill memory, testing each new region
    held_chunks = {gpu_id: [] for gpu_id in gpu_ids}  # keep alive
    overall_failed = 0
    overall_passed = 0
    region_results = []

    for i in range(n_regions):
        region_start = i * sweep_chunk_gb
        region_end = (i + 1) * sweep_chunk_gb

        logger.info(
            f"--- Sweep region {i + 1}/{n_regions} "
            f"(~{region_start:.0f}-{region_end:.0f} GB offset) ---"
        )

        # Run operator tests — tensors land in the next free region
        detector = CrossGPUDetector(
            gpu_ids=gpu_ids,
            test_sizes=test_sizes,
            hash_inputs=hash_inputs,
        )

        result = detector.run(
            categories=categories,
            dtypes=dtypes,
            iterations=iterations,
            run_pipelines=run_pipelines,
            fail_fast=args.fail_fast,
            verbose=args.verbose,
            precision_modes=args.precision_modes,
            duration_limit=args.duration_limit,
        )

        overall_passed += result["passed"]
        overall_failed += result["failed"]

        status = "FAIL" if result["failed"] > 0 else "PASS"
        region_results.append(
            (i, region_start, region_end, status, result["passed"], result["failed"])
        )

        if result.get("time_limited"):
            logger.info(f"  Region {i + 1}/{n_regions}: duration limit reached")
            break

        if result["failed"] > 0:
            logger.error(
                f"  Region {i + 1}/{n_regions}: SDC DETECTED "
                f"({result['failed']} failures)"
            )
            if args.fail_fast:
                break
        else:
            logger.info(
                f"  Region {i + 1}/{n_regions}: PASSED "
                f"({result['passed']} comparisons clean)"
            )

        # Fill this region on all GPUs so next iteration's tensors
        # land in the next region
        for gpu_id in gpu_ids:
            device = f"cuda:{gpu_id}"
            try:
                t = torch.randn(chunk_elements, device=device)
                held_chunks[gpu_id].append(t)
            except torch.cuda.OutOfMemoryError:
                # Out of memory — we've tested all available regions
                logger.info(f"  GPU {gpu_id}: OOM at region {i + 1}, ending sweep")
                break
        else:
            # Only continue if all GPUs allocated successfully
            continue
        break  # OOM on at least one GPU

    # Summary
    actual_regions = len(region_results)
    logger.info(f"\n{'=' * 60}")
    logger.info("SWEEP REGION SUMMARY:")
    for idx, start, end, status, passed, failed in region_results:
        marker = "FAIL" if status == "FAIL" else "PASS"
        logger.info(
            f"  Region {idx + 1} (~{start:.0f}-{end:.0f} GB): {marker} "
            f"({passed} passed, {failed} failed)"
        )

    if overall_failed > 0:
        logger.error(
            f"\nSWEEP SDC DETECTED: {overall_failed} failures across "
            f"{actual_regions} memory regions ({overall_passed} passed)"
        )
    else:
        logger.info(
            f"\nSWEEP PASSED: {overall_passed} comparisons clean across "
            f"{actual_regions} memory regions"
        )
    logger.info(f"{'=' * 60}")

    print(
        f"\nResult: {{'sweep_regions': {actual_regions}, "
        f"'total_passed': {overall_passed}, "
        f"'total_failed': {overall_failed}}}"
    )

    # Cleanup
    del held_chunks
    torch.cuda.empty_cache()

    return 1 if overall_failed > 0 else 0


def _run_self_mode(args):
    """Run self-consistency testing on each GPU in parallel."""
    from concurrent.futures import as_completed, ThreadPoolExecutor

    from .detector import GPUSDCDetector

    gpu_ids = _get_gpu_ids()
    test_sizes = _parse_sizes(args.sizes) if args.sizes else None

    if args.list_ops:
        detector = GPUSDCDetector(test_sizes=test_sizes)
        detector.list_ops()
        return 0

    categories = None
    if args.categories:
        categories = [c.strip() for c in args.categories.split(",")]

    dtypes = [d.strip() for d in args.dtypes.split(",")]

    def _test_gpu(gpu_id):
        detector = GPUSDCDetector(
            gpu_id=gpu_id,
            num_iterations=args.iterations,
            test_sizes=test_sizes,
        )
        try:
            result = detector.run_all_tests(
                categories=categories,
                dtypes=dtypes,
                run_pipelines=args.pipelines,
            )
            result["gpu_id"] = gpu_id
            return result
        except (ValueError, RuntimeError) as e:
            return {"gpu_id": gpu_id, "error": str(e)}

    all_results = []
    overall_rc = 0

    if len(gpu_ids) == 1:
        # Single GPU — no threading overhead
        all_results.append(_test_gpu(gpu_ids[0]))
    else:
        logger.info(
            f"Running self-consistency tests on {len(gpu_ids)} GPUs in parallel..."
        )
        with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
            futures = {executor.submit(_test_gpu, g): g for g in gpu_ids}
            for future in as_completed(futures):
                all_results.append(future.result())

    # Sort by GPU ID for consistent output
    all_results.sort(key=lambda r: r["gpu_id"])

    for r in all_results:
        if "error" in r:
            print(f"ERROR on GPU {r['gpu_id']}: {r['error']}", file=sys.stderr)
            overall_rc = 1
        else:
            if r["ops_failed"] > 0:
                overall_rc = 1

    if len(gpu_ids) > 1:
        logger.info(f"\n{'=' * 60}")
        logger.info("SELF-CONSISTENCY SUMMARY")
        logger.info(f"{'=' * 60}")
        for r in all_results:
            if "error" in r:
                logger.error(f"  GPU {r['gpu_id']}: ERROR ({r['error']})")
            else:
                status = "FAIL" if r["ops_failed"] > 0 else "PASS"
                logger.info(
                    f"  GPU {r['gpu_id']}: {status} "
                    f"({r['ops_passed']}/{r['ops_total']} passed)"
                )

    for r in all_results:
        print(f"\nResult (GPU {r['gpu_id']}): {r}")

    return overall_rc


def _run_cross_gpu_mode(args, preset=None):
    """Run cross-GPU operator comparison."""
    from .cross_gpu import CrossGPUDetector

    # Apply preset if specified, with explicit args overriding
    iterations = args.iterations
    test_sizes = _parse_sizes(args.sizes) if args.sizes else None
    categories = None
    if args.categories:
        categories = [c.strip() for c in args.categories.split(",")]
    hash_inputs = args.hash_inputs
    run_pipelines = args.pipelines

    if preset:
        cfg = PRESETS[preset]
        if args.iterations == 1:  # default, not explicitly set
            iterations = cfg["iterations"]
        if test_sizes is None:
            test_sizes = cfg["sizes"]
        if categories is None:
            categories = cfg["categories"]
        hash_inputs = hash_inputs or cfg["hash_inputs"]
        run_pipelines = run_pipelines or cfg["pipelines"]

    # Determine GPUs
    gpu_ids = _get_gpu_ids()

    if len(gpu_ids) < 2:
        if len(gpu_ids) == 0:
            logger.error("no GPUs detected")
            return 1
        # Single-GPU platform (e.g. Grace Hopper GH200) — cross-GPU comparison
        # is impossible, fall back to self-consistency mode so the GPU still
        # gets tested and we don't pollute SDC metrics with false failures.
        logger.info(
            "Single GPU detected — falling back to self-consistency mode "
            "(cross-GPU comparison requires 2+ GPUs)"
        )
        # Apply preset-enhanced values so self-mode runs at the configured depth.
        args.iterations = iterations
        if test_sizes is not None:
            args.sizes = ",".join(f"{w}x{h}" for w, h in test_sizes)
        if categories is not None:
            args.categories = ",".join(categories)
        args.hash_inputs = hash_inputs
        args.pipelines = run_pipelines
        return _run_self_mode(args)

    dtypes = [d.strip() for d in args.dtypes.split(",")]

    # Random warmup: vary which HBM regions are tested each run
    _held_tensors = _random_warmup(gpu_ids, test_sizes or [(4096, 4096)])

    detector = CrossGPUDetector(
        gpu_ids=gpu_ids,
        test_sizes=test_sizes,
        hash_inputs=hash_inputs,
    )

    result = detector.run(
        categories=categories,
        dtypes=dtypes,
        iterations=iterations,
        run_pipelines=run_pipelines,
        fail_fast=args.fail_fast,
        verbose=args.verbose,
        precision_modes=args.precision_modes,
        duration_limit=args.duration_limit,
    )

    # Release warmup tensors and return memory to CUDA runtime
    import torch

    del _held_tensors
    torch.cuda.empty_cache()

    print(f"\nResult: {result}")

    return 1 if result["failed"] > 0 else 0


def _resolve_precision_modes(args, parser):
    """Validate and resolve precision mode arguments.

    Handles --precision-modes, --randomize-precision, and --no-randomize-precision
    interactions. When no precision mode is specified, randomizes by default.
    """
    from .precision import VALID_PRECISION_MODES

    if args.precision_modes is not None:
        for pm in args.precision_modes:
            if pm not in VALID_PRECISION_MODES:
                parser.error(
                    f"Invalid precision mode '{pm}'. Valid: {sorted(VALID_PRECISION_MODES)}"
                )

    if args.precision_modes is not None and args.randomize_precision is not True:
        # User explicitly chose precision modes — honor them
        pass
    elif args.randomize_precision is False:
        # User explicitly disabled randomization
        args.precision_modes = ["fp32"]
    else:
        # Default behavior: randomize
        from .randomize import randomize_precision

        args.precision_modes = randomize_precision(
            current_modes=args.precision_modes or ["fp32"],
            pool=args.precision_modes,
        )


def _resolve_mode_preset(
    mode: str | None, preset: str | None
) -> tuple[str, str | None]:
    """Resolve the effective (mode, preset) pair from CLI args.

    Honor an explicit --mode even when --preset is also set so callers can
    compose them (e.g., --mode random --preset standard). --preset alone
    implies cross-gpu for back-compat.
    """
    if mode is None:
        mode = "cross-gpu" if preset else "self"
    return mode, preset


def main():
    parser = argparse.ArgumentParser(
        description="PyTorch operator-level SDC detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  %(prog)s --preset quick                              Quick SDC check, all GPUs (~1-2 min)
  %(prog)s --preset standard                           Standard check, all GPUs (~5 min)
  %(prog)s --preset thorough                           Full diagnosis + sweep (~25-30 min)
  %(prog)s --preset quick --fail-fast                  Stop on first failure
  %(prog)s --preset quick -v                           Full detail on failures
  %(prog)s --mode self                                 Self-consistency, all GPUs
  %(prog)s --mode cross-gpu                            Cross-GPU, all GPUs
  %(prog)s --mode sweep                                Systematic full-HBM sweep
  %(prog)s --list-ops                                  List all registered ops
""",
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=["self", "cross-gpu", "sweep"],
        default=None,
        help="Test mode: self (single-GPU), cross-gpu (compare across GPUs), "
        "sweep (systematic HBM region coverage). "
        "Default: self",
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=list(PRESETS.keys()),
        default=None,
        help="Predefined configuration (implies --mode cross-gpu). "
        "quick (~1-2 min), standard (~5 min), thorough (~25-30 min, includes sweep)",
    )

    # Op configuration
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Iterations per op (default: 1 for cross-gpu, 3 for self)",
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default=None,
        help="Comma-separated tensor sizes (e.g. 4096x4096,14000x14000)",
    )
    parser.add_argument(
        "--categories",
        type=str,
        default=None,
        help="Comma-separated op categories (default: all)",
    )
    parser.add_argument(
        "--dtypes",
        type=str,
        default="float32",
        help="Comma-separated dtypes: float32, float16, bfloat16 (default: float32)",
    )
    parser.add_argument(
        "--pipelines",
        action="store_true",
        default=False,
        help="Also run composite pipeline tests",
    )

    # Cross-GPU specific
    parser.add_argument(
        "--hash-inputs",
        action="store_true",
        default=False,
        help="Hash input tensors to distinguish memory vs compute corruption "
        "(cross-gpu mode only)",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        default=False,
        help="Stop on first failure (default: run all ops to completion "
        "and print per-GPU summary).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Print full element-level diffs, byte offsets, sample values "
        "on failure. Combine with --fail-fast for detailed single-failure output.",
    )

    # Precision modes
    parser.add_argument(
        "--precision-modes",
        type=str,
        nargs="+",
        default=None,
        help="Hardware precision modes to test. Each exercises a different "
        "datapath: fp32 (vector ALUs), tf32 (matrix cores), fp16, bf16, fp8. "
        "Multiple modes run as additive passes. "
        "When not specified, one mode is randomly selected per run "
        "(see --no-randomize-precision to disable). "
        "Example: --precision-modes fp32 tf32",
    )
    randomize_group = parser.add_mutually_exclusive_group()
    randomize_group.add_argument(
        "--randomize-precision",
        action="store_true",
        default=None,
        dest="randomize_precision",
        help="Randomly select one precision mode per run from the pool "
        "(fp32, tf32, fp16, bf16). This is the default when "
        "--precision-modes is not specified.",
    )
    randomize_group.add_argument(
        "--no-randomize-precision",
        action="store_false",
        default=None,
        dest="randomize_precision",
        help="Disable precision randomization. Uses fp32 if "
        "--precision-modes is not specified.",
    )

    # Sweep mode specific
    parser.add_argument(
        "--sweep-chunk-gb",
        type=float,
        default=4.0,
        help="Chunk size for sweep mode in GB (default: 4.0). "
        "Smaller = more regions tested but slower.",
    )

    # Utility
    parser.add_argument(
        "--duration-limit",
        type=int,
        default=None,
        help="Hard duration limit in seconds. Uses SIGALRM to interrupt "
        "immediately — no waiting for the current op or GPU kernel to finish. "
        "Reports all results collected before the cutoff. "
        "Useful for capping run time in automated pipelines.",
    )
    parser.add_argument(
        "--list-ops",
        "--list_ops",
        action="store_true",
        default=False,
        dest="list_ops",
        help="List all registered operations and exit",
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Set CUBLAS workspace config for deterministic mode
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    _resolve_precision_modes(args, parser)

    mode, preset = _resolve_mode_preset(args.mode, args.preset)

    # Handle --list-ops before mode dispatch
    if args.list_ops:
        from .detector import GPUSDCDetector

        detector = GPUSDCDetector()
        detector.list_ops()
        return 0

    # Set self-mode default iterations if not explicitly changed
    if mode == "self" and args.iterations == 1:
        args.iterations = 3

    # Dispatch
    ran_cross_gpu = False
    if mode == "self":
        rc = _run_self_mode(args)
    elif mode == "cross-gpu":
        rc = _run_cross_gpu_mode(args, preset=preset)
        ran_cross_gpu = True
    elif mode == "sweep":
        rc = _sweep_cross_gpu(args, preset=preset)
    else:
        print(f"Unknown mode: {mode}", file=sys.stderr)
        rc = 1

    # If preset=thorough, also run sweep for full HBM coverage.
    # Only applies after cross-gpu mode: sweep is a
    # cross-GPU comparison and is meaningless after self/sweep modes.
    # Skip sweep if fail-fast already caught SDC in cross-gpu phase.
    if preset == "thorough" and ran_cross_gpu and not (rc != 0 and args.fail_fast):
        logger.info("\n=== Running full-HBM sweep (thorough preset) ===")
        sweep_chunk_gb = PRESETS["thorough"]["sweep_chunk_gb"]
        args.sweep_chunk_gb = sweep_chunk_gb
        # Use minimal config per region — sweep is for coverage, not depth
        args.iterations = 1
        args.categories = "backward_pass"
        args.hash_inputs = True
        args.pipelines = False
        sweep_rc = _sweep_cross_gpu(args, preset=None)
        rc = max(rc, sweep_rc)

    sys.exit(rc)


if __name__ == "__main__":
    main()
