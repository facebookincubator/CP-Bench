# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import re
from collections import defaultdict


def is_speed_within_percentage(value, target, percentage_threshold):
    """Check if a value is within a certain percentage of a target."""
    return value < target * (1 + percentage_threshold / 100)


def is_thruput_within_percentage(value, target, percentage_threshold):
    """Check if a value is within a certain percentage of a target."""
    return value > target * (1 - percentage_threshold / 100)


def check_if_distributed_stress_testing_pass(lines):
    """Check if distributed stress testing pass."""
    
    print(
        "---- Stress test looks at the functional reliability, hard crash errors, etc, e.g., CUDA IMA, HBM uncorrectable error, etc -------"
    )
    all_return_codes_zero = True
    for line in lines:
        if "return code: 0" not in line and "return code:" in line:
            all_return_codes_zero = False
            break
    if all_return_codes_zero:
        print("PASS: Distributed Stress Testing.")
        return 1
    else:
        print("FAIL: Distributed Stress Testing.")
        return 0


def check_if_concurrent_stress_testing_pass(lines):
    """Check if concurrent stress testing pass."""
    
    all_return_codes_zero = True
    for line in lines:
        if "return code: 0" not in line and "return code:" in line:
            all_return_codes_zero = False
            break
    if all_return_codes_zero:
        print("PASS: Concurrent stress testing.")
        return 1
    else:
        print("FAIL: Concurrent stress testing.")
        return 0


def check_if_distributed_perf_pass(lines, target_step_time, percentage_threshold):
    """Check if distributed performance pass by comparing against a golden reference."""

    print(
        "----- Perf test looks at step time (defined as time need to finish a single step, i.e., how fast does it train, on this given model) -----"
    )
    pattern = re.compile(
        r"gpu_device: distributed, return code: (\d+), result: \{'return_code': \[(\d+)\], 'fp32_train_step_time': \[([\d.]+)\], 'fp32_train_throughput': \[([\d.]+)\]\}"
    )
    for line in lines:
        match = pattern.search(line)
        if match:
            return_code = int(match.group(1))
            fp32_train_step_time = float(match.group(3))

            if return_code == 0 and is_speed_within_percentage(
                fp32_train_step_time, target_step_time, percentage_threshold
            ):
                print(
                    f"PASS (distributed perf): Actual perf {fp32_train_step_time}, target perf {target_step_time}, diff <= threshold {percentage_threshold}%"
                )
            else:
                print(
                    f"WARNING (distributed perf): Actual perf {fp32_train_step_time}, target perf {target_step_time}, diff > threshold {percentage_threshold}%",
                )


def check_if_concurrent_perf_pass(lines, target_step_time, percentage_threshold):
    """Process lines for concurrent performance mode."""
    pattern = re.compile(
        r"gpu_device: (\d+), return code: (\d+), result: \{'return_code': \[(\d+)\], 'fp32_train_step_time': \[([\d.]+)\], 'fp32_train_throughput': \[([\d.]+)\]\}"
    )
    for line in lines:
        match = pattern.search(line)
        if match:
            gpu_device = int(match.group(1))
            return_code = int(match.group(2))
            fp32_train_step_time = float(match.group(4))

            if return_code == 0 and is_speed_within_percentage(
                fp32_train_step_time, target_step_time, percentage_threshold
            ):
                print(
                    f"PASS (individual_gpu perf): Actual perf {fp32_train_step_time}, target perf {target_step_time}, diff <= threshold {percentage_threshold}% for GPU device {gpu_device}."
                )
            else:
                print(
                    f"WARNING (individual_gpu perf): Actual perf {fp32_train_step_time}, target perf {target_step_time}, diff > threshold {percentage_threshold}% for GPU device {gpu_device}."
                )


def parse_checksums(file_lines):
    """Parse checksum values from log lines."""
    pattern = re.compile(r"Checksum at step (\d+): ([\d.]+) at GPU rank \d+")
    checksums = defaultdict(set)
    for line in file_lines:
        match = pattern.search(line)
        if match:
            step = int(match.group(1))
            checksum = float(match.group(2))
            checksums[step].add(checksum)
    return checksums


def check_if_distributed_sdc_pass(lines1, lines2):
    """Check if distributed sdc pass. Need to compare host vs. host"""

    print(
        "------ SDC test looks at if there is silent data corruption in the model running ----- "
    )
    checksums1 = parse_checksums(lines1)
    checksums2 = parse_checksums(lines2)
    common_steps = set(checksums1.keys()).intersection(set(checksums2.keys()))
    for step in sorted(common_steps):
        if checksums1[step] == checksums2[step]:
            pass
        else:
            print(f"FAIL: Distributed SDC testing at step {step}.")
            return
    print("PASS: Distributed SDC testing.")


def check_if_concurrent_sdc_pass(lines):
    """Check if the checksum values at each step are the same across all GPU ranks."""
    pattern = re.compile(r"Checksum at step (\d+): ([\d.]+) at GPU rank (\d+)")
    checksums_by_step = defaultdict(set)
    for line in lines:
        match = pattern.search(line)
        if match:
            step = int(match.group(1))
            checksum = float(match.group(2))
            checksums_by_step[step].add(checksum)
    for step, checksums in checksums_by_step.items():
        if len(checksums) == 1:
            pass  # Do nothing if checksums are consistent
        else:
            print(f"Step {step}: Checksum values are inconsistent across GPU ranks.")
            print("FAIL: Concurrent SDC testing.")
            return
    print("PASS: Concurrent SDC testing.")


def main(
    distributed_target_step_time=None,
    concurrent_target_step_time=None,
    percentage_threshold=None,
    ref_log_file=None,
):
    """Main function."""
    print(
        "*************************************************************************************************"
    )
    print(
        "*************************************************************************************************"
    )
    print(
        "******************************** Perform Testing Analysis ***************************************"
    )
    print(
        "*************************************************************************************************"
    )
    print(
        "*************************************************************************************************"
    )
    with open("run_distributed.log", "r") as file:
        distributed_log_lines = file.readlines()
    with open("run_concurrent.log", "r") as file:
        concurrent_log_lines = file.readlines()
    dist_pass = check_if_distributed_stress_testing_pass(distributed_log_lines)
    conc_pass = check_if_concurrent_stress_testing_pass(concurrent_log_lines)
    if dist_pass == 1 and distributed_target_step_time:
        check_if_distributed_perf_pass(
            distributed_log_lines,
            target_step_time=distributed_target_step_time,
            percentage_threshold=percentage_threshold,
        )
    if conc_pass == 1 and concurrent_target_step_time:
        check_if_concurrent_perf_pass(
            concurrent_log_lines,
            target_step_time=concurrent_target_step_time,
            percentage_threshold=percentage_threshold,
        )
    if (
        dist_pass == 1 and ref_log_file
    ):  # Check SDC only if a reference log file is provided
        with open(ref_log_file, "r") as file:
            ref_log_lines = file.readlines()
        check_if_distributed_sdc_pass(distributed_log_lines, ref_log_lines)
    if conc_pass == 1:
        check_if_concurrent_sdc_pass(concurrent_log_lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform testing analysis.")
    parser = argparse.ArgumentParser(description="Perform testing analysis.")
    parser.add_argument(
        "--distributed_target_step_time",
        type=float,
        help="Target step time for distributed performance check.",
    )
    parser.add_argument(
        "--concurrent_target_step_time",
        type=float,
        help="Target step time for concurrent performance check.",
    )
    parser.add_argument(
        "--percentage_threshold",
        type=float,
        help="Percentage threshold for performance check.",
    )
    parser.add_argument(
        "--ref_log_file",
        type=str,
        help="Path to the reference log file.",
    )
    args = parser.parse_args()
    main(
        args.distributed_target_step_time,
        args.concurrent_target_step_time,
        args.percentage_threshold,
        args.ref_log_file,
    )
