# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse

from benchmarks.context import Framework, Platform
from benchmarks.logging import logger
from benchmarks.registry import BenchmarkRegistry

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--distributed",
        action="store_true",
        default=False,
        help="Whether to enable distributed training.",
    )
    parser.add_argument(
        "--gpu_rank",
        type=int,
        default=0,
        required=False,
        help="which gpu rank to use.",
    )
    parser.add_argument(
        "--run_count",
        type=int,
        default=1,
        required=False,
        help="how many times to run.",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=120,
        required=False,
        help="how long duration to run.",
    )
    parser.add_argument(
        "--sample_count",
        type=int,
        default=1024,
        required=False,
        help="The number of data samples in dataset.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=2048,
        required=False,
        help="The number of training/testing steps.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        required=False,
        help="The number of data samples in a batch.",
    )
    parser.add_argument(
        "--precision",
        default="float32",
        required=False,
        help="model precision: float32, float16",
    )
    parser.add_argument(
        "--num_warmup",
        type=int,
        default=10,
        required=False,
        help="number of warmup steps.",
    )
    parser.add_argument(
        "--sdc_check",
        type=int,
        default=0,
        required=False,
        help="whether to sdc check or not.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
        required=False,
        help="random seed for randomness.",
    )
    args = parser.parse_args()

    # Specify the model name and benchmark parameters.
    model_name = "llama-large"
    parameters = "--seq_len 256"
    if args.gpu_rank in range(0, 8):
        parameters += " --gpu_device " + str(args.gpu_rank)
        print("gpu_device: ", args.gpu_rank)
        target_gpu = args.gpu_rank
    if args.distributed:
        parameters += " --distributed_impl ddp --distributed_backend nccl"
        target_gpu = "distributed"
        print("Distributed Training")
    if args.batch_size:
        parameters += " --batch_size " + str(args.batch_size)
    if args.precision:
        parameters += " --precision " + args.precision
    if args.num_warmup:
        parameters += " --num_warmup " + str(args.num_warmup)
    if args.run_count:
        parameters += " --run_count " + str(args.run_count)
    if args.duration:
        parameters += " --duration " + str(args.duration)
    if args.sample_count:
        parameters += " --sample_count " + str(args.sample_count)
    if args.num_steps:
        parameters += " --num_steps " + str(args.num_steps)
    if args.sdc_check:
        parameters += " --sdc_check " + str(args.sdc_check)
    if args.random_seed:
        parameters += " --random_seed " + str(args.random_seed)

    # Create context for llama-large benchmark and run it for 120 * 2 seconds.
    context = BenchmarkRegistry.create_benchmark_context(
        model_name,
        platform=Platform.CUDA,
        parameters=parameters,
        framework=Framework.PYTORCH,
    )

    benchmark = BenchmarkRegistry.launch_benchmark(context)
    if benchmark:
        logger.info(
            "benchmark: {}, gpu_device: {}, return code: {}, result: {}".format(
                benchmark.name, target_gpu, benchmark.return_code, benchmark.result
            )
        )
