# Copyright (c) Meta Platforms, Inc. and affiliates.

import random
import shutil
import subprocess

import time
from concurrent.futures import ThreadPoolExecutor

from threading import Event

import click

from common.accelerator import AcceleratorVendor


def can_monitor() -> bool:
    if AcceleratorVendor.get() != AcceleratorVendor.NVIDIA:
        click.echo(
            f"WARNING: Detected Vendor {AcceleratorVendor.get()}: --monitoring-enabled only support for nvidia GPU."
        )
        return False
        if shutil.which("nvidia-smi") is None:
            click.echo(
                "WARNING: 'nvidia-smi' not found. Cannot monitor GPU utilization."
            )
            return False
    return True


def monitor_accelerator(interval: float, stop_event: Event) -> None:
    while not stop_event.is_set():
        try:
            match AcceleratorVendor.get():
                case AcceleratorVendor.NVIDIA:
                    output = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=utilization.gpu",
                            "--format=csv,noheader,nounits",
                        ],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                        text=True,
                        check=True,  # Raises CalledProcessError if command fails
                    )
                case _:
                    click.echo("Monitoring is not implemented for this device")
                    return

            util_lines = output.stdout.strip().split("\n")
            util_vals = [line.strip() for line in util_lines if line]
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            click.echo(f"{timestamp} | GPU Utilization: {' '.join(util_vals)}%")

        except subprocess.CalledProcessError:
            click.echo(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} | ERROR: Failed to retrieve GPU utilization.",
                err=True,
            )
            break

        stop_event.wait(
            interval
        )  # Waits for the next cycle but stops if the event is set


def _validate_dist_args(ctx, param, value):
    if ctx.params["mode"] != "distributed" and value != param.default:
        raise click.BadParameter(
            f"The {param.name} option is only valid in distributed mode"
        )
    return value


@click.command()
@click.option(
    "--mode",
    type=click.Choice(["concurrent", "distributed", "sequential"]),
    required=True,
    help="Mode to run the benchmarks: 'concurrent', 'distributed', 'sequential'.",
)
@click.option(
    "--nnodes",
    type=int,
    default=1,
    help="Number of nodes for distributed mode",
    callback=_validate_dist_args,
)
@click.option(
    "--rdzv-backend",
    type=str,
    default="c10d",
    help="Rendezvous backend",
    callback=_validate_dist_args,
)
@click.option(
    "--rdzv-endpoint",
    type=str,
    default="localhost:29600",
    help="Rendezvous endpoint",
    callback=_validate_dist_args,
)
@click.option(
    "--rdzv-id",
    type=str,
    default="20",
    help="rendezvous id",
    callback=_validate_dist_args,
)
@click.option(
    "--accelerator-vendor",
    type=click.Choice(
        [str(v.name) for v in AcceleratorVendor.__members__.values()],
        case_sensitive=False,
    ),
    default=None,
    help="Accelerator vendor to use. Auto-detection only works for NVIDIA GPUs.",
)
@click.option(
    "--start-gpu",
    default=0,
    help="Starting GPU device number (only for sequential/concurrent mode).",
)
@click.option(
    "--end-gpu",
    default=7,
    help="Ending GPU device number (only for sequential/concurrent mode).",
)
@click.option("--run-count", default=1, help="Number of runs per GPU device.")
@click.option("--duration", default=120, help="Number of seconds to run.")
@click.option("--sample_count", default=1024, help="Number of data samples.")
@click.option("--num_steps", default=2048, help="Number of training/testing steps.")
@click.option("--num_warmup", default=10, help="number of warm up steps.")
@click.option("--batch_size", default=32, help="number of data samples in a batch.")
@click.option("--sdc_check", default=0, help="whether perform SDC check or not.")
@click.option("--random_seed", default=0, help="the random seed for randomness.")
@click.option(
    "--monitoring-enabled",
    is_flag=True,
    help="enable monitoring loop to capture SM utilization etc",
)
@click.option(
    "--monitoring-interval",
    default=1.0,
    help="emonitoring interval in seconds (default: 1.0s)",
)
@click.option(
    "--precision", default="float32", help="model precision: float16 or float32."
)
@click.option(
    "--models",
    default="random",
    help='Comma-separated list of models to run (e.g., "alexnet,resnet,lstm,bert,gpt2,llama", or "random" which randomly select one model to run).',
)
def run_benchmarks(
    mode,
    nnodes,
    rdzv_backend,
    rdzv_endpoint,
    rdzv_id,
    accelerator_vendor,
    start_gpu,
    end_gpu,
    batch_size,
    num_warmup,
    precision,
    duration,
    run_count,
    sample_count,
    num_steps,
    models,
    sdc_check,
    random_seed,
    monitoring_enabled,
    monitoring_interval,
):
    """Run specified benchmarks on multiple GPU devices either concurrently, distributedly, or sequentially."""
    AcceleratorVendor.set(accelerator_vendor)
    monitor_executor = None
    stop_event = Event()
    monitor_future = None
    if monitoring_enabled and can_monitor():
        monitor_executor = ThreadPoolExecutor(max_workers=1)
        monitor_future = monitor_executor.submit(
            monitor_accelerator, monitoring_interval, stop_event
        )

    # Determine models to run
    if models == "random":
        models_to_run = [
            random.choice(["alexnet", "resnet", "lstm", "bert", "gpt2", "llama"])
        ]
    else:
        models_to_run = models.split(",")

    model_scripts = {
        "alexnet": "pytorch_alexnet.py",
        "resnet": "pytorch_resnet.py",
        "lstm": "pytorch_lstm.py",
        "bert": "pytorch_bert_large.py",
        "gpt2": "pytorch_gpt2_large.py",
        "llama": "pytorch_llama.py",
    }

    if mode == "concurrent":

        def run_model_on_gpu(gpu_device):
            for model in models_to_run:
                if model in model_scripts:
                    script = model_scripts[model]
                    click.echo(
                        f"Running {model} with batch size of {batch_size} and precision of {precision} on GPU device {gpu_device} for a lesser of {duration} seconds or {num_steps} steps"
                    )
                    subprocess.run(
                        [
                            "python3",
                            script,
                            "--gpu_rank",
                            str(gpu_device),
                            "--duration",
                            str(duration),
                            "--run_count",
                            str(run_count),
                            "--sample_count",
                            str(sample_count),
                            "--num_steps",
                            str(num_steps),
                            "--num_warmup",
                            str(num_warmup),
                            "--batch_size",
                            str(batch_size),
                            "--precision",
                            precision,
                            "--sdc_check",
                            str(sdc_check),
                            "--random_seed",
                            str(random_seed),
                        ]
                    )
                else:
                    click.echo(f"Model {model} not recognized. Skipping...")

        # Create a thread pool executor to run tasks concurrently
        with ThreadPoolExecutor(max_workers=end_gpu - start_gpu + 1) as executor:
            executor.map(run_model_on_gpu, range(start_gpu, end_gpu + 1))

    elif mode == "distributed":
        for model in models_to_run:
            if model in model_scripts:
                script = model_scripts[model]
                click.echo(
                    f"Running {model} with batch size of {batch_size} and precision of {precision} for distributed training for a lesser of {duration} seconds or {num_steps} steps"
                )
                subprocess.run(
                    [
                        "torchrun",
                        "--nnodes",
                        str(nnodes),
                        "--nproc_per_node=8",
                        "--rdzv_backend",
                        rdzv_backend,
                        "--rdzv_endpoint",
                        rdzv_endpoint,
                        "--rdzv_id",
                        rdzv_id,
                        script,
                        "--distributed",
                        "--duration",
                        str(duration),
                        "--run_count",
                        str(run_count),
                        "--sample_count",
                        str(sample_count),
                        "--num_steps",
                        str(num_steps),
                        "--num_warmup",
                        str(num_warmup),
                        "--batch_size",
                        str(batch_size),
                        "--precision",
                        precision,
                        "--sdc_check",
                        str(sdc_check),
                        "--random_seed",
                        str(random_seed),
                    ]
                )
            else:
                click.echo(f"Model {model} not recognized. Skipping...")

    elif mode == "sequential":
        for model in models_to_run:
            if model in model_scripts:
                script = model_scripts[model]
                click.echo(
                    f"Running {model} with batch size of {batch_size} and precision of {precision} sequentially for a lesser of {duration} seconds or {num_steps} steps"
                )
                subprocess.run(
                    [
                        "python3",
                        script,
                        "--gpu_rank",
                        str(start_gpu),
                        "--duration",
                        str(duration),
                        "--run_count",
                        str(run_count),
                        "--sample_count",
                        str(sample_count),
                        "--num_steps",
                        str(num_steps),
                        "--num_warmup",
                        str(num_warmup),
                        "--batch_size",
                        str(batch_size),
                        "--precision",
                        precision,
                        "--sdc_check",
                        str(sdc_check),
                        "--random_seed",
                        str(random_seed),
                    ]
                )

    if monitor_executor is not None:
        stop_event.set()
        monitor_future.result()
        monitor_executor.shutdown(wait=True)
        click.echo("Monitoring thread shut down cleanly.")

    click.echo("All tasks are completed.")


def main():
    run_benchmarks()


if __name__ == "__main__":
    main()
