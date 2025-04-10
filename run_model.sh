# Copyright (c) Meta Platforms, Inc. and affiliates.

#!/bin/bash

# Export environment variables
export NCCL_DEBUG=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Get the current working directory
CWD=$PWD

# execute from anywhere
cd "$(dirname "$0")"

# Set default batch sizes and target perf number (based on H100_80GB) and duration
DURATION=1800
BATCH_SIZE_DISTRIBUTED=24
BATCH_SIZE_CONCURRENT=28
DISTRIBUTED_TARGET_STEP_TIME=477.11
CONCURRENT_TARGET_STEP_TIME=513.18


while [[ $# -gt 0 ]]; do
  case $1 in
    -t)
      DURATION="$2"
      shift 2
      ;;
    *)
      echo "Invalid option $1"
      exit 1
      ;;
  esac
done



# collect GPU information
nvidia-smi --query-gpu=name,pci.bus_id,gpu_uuid,gpu_serial,vbios_version,power.limit,clocks.current.graphics --format=csv

# store information about gpus to be used during initialization
gpu_info=$(rgpu getGpuInfo)

# 96G H100 memory.total 102625181696
if jq -e '(.gpu_info.gpu_details[0].detail.nvidia_info.memory.total == 102625181696)' <<< "$gpu_info" > /dev/null; then
  BATCH_SIZE_DISTRIBUTED=32
  BATCH_SIZE_CONCURRENT=36
  DISTRIBUTED_TARGET_STEP_TIME=735
  CONCURRENT_TARGET_STEP_TIME=749
  echo "Found 96G H100, set distributed batch size to $BATCH_SIZE_DISTRIBUTED, and concurrent batch size to $BATCH_SIZE_CONCURRENT"
  python run_benchmarks.py --mode distributed --batch_size $BATCH_SIZE_DISTRIBUTED --models 'llama' --precision 'float32' --sdc_check 1 --random_seed 1 --duration $DURATION --num_steps 1000000 | tee run_distributed.log
  python run_benchmarks.py --mode concurrent --batch_size $BATCH_SIZE_CONCURRENT --models 'llama' --precision 'float32' --sdc_check 1 --random_seed 1 --duration $DURATION --num_steps 1000000 | tee run_concurrent.log
  python analytics_gpu.py --distributed_target_step_time $DISTRIBUTED_TARGET_STEP_TIME --concurrent_target_step_time $CONCURRENT_TARGET_STEP_TIME --percentage_threshold 3 --ref_log_file ./ref_log/ref_gtt_h100_96gb_dist.txt | tee "$CWD"/analytics_gpudiag.log
# 80G A100 memory.total 85899345920
elif jq -e '(.gpu_info.gpu_details[0].detail.nvidia_info.memory.total == 85899345920)' <<< "$gpu_info" > /dev/null; then
  BATCH_SIZE_DISTRIBUTED=24
  BATCH_SIZE_CONCURRENT=28
  DISTRIBUTED_TARGET_STEP_TIME=1021
  CONCURRENT_TARGET_STEP_TIME=1110
  echo "Found 80G A100, set distributed batch size to $BATCH_SIZE_DISTRIBUTED, and concurrent batch size to $BATCH_SIZE_CONCURRENT"
  python run_benchmarks.py --mode distributed --batch_size $BATCH_SIZE_DISTRIBUTED --models 'llama' --precision 'float32' --sdc_check 1 --random_seed 1 --duration $DURATION --num_steps 1000000 | tee run_distributed.log
  python run_benchmarks.py --mode concurrent --batch_size $BATCH_SIZE_CONCURRENT --models 'llama' --precision 'float32' --sdc_check 1 --random_seed 1 --duration $DURATION --num_steps 1000000 | tee run_concurrent.log
  python analytics_gpu.py --distributed_target_step_time $DISTRIBUTED_TARGET_STEP_TIME --concurrent_target_step_time $CONCURRENT_TARGET_STEP_TIME --percentage_threshold 3 --ref_log_file ./ref_log/ref_zion_a100_80gb_dist.txt | tee "$CWD"/analytics_gpudiag.log
# 80G H100 memory.total 85520809984
elif jq -e '(.gpu_info.gpu_details[0].detail.nvidia_info.memory.total == 85520809984)' <<< "$gpu_info" > /dev/null; then
  BATCH_SIZE_DISTRIBUTED=24
  BATCH_SIZE_CONCURRENT=28
  DISTRIBUTED_TARGET_STEP_TIME=477
  CONCURRENT_TARGET_STEP_TIME=513
  echo "Found 80G H100, set distributed batch size to $BATCH_SIZE_DISTRIBUTED, and concurrent batch size to $BATCH_SIZE_CONCURRENT"
  python run_benchmarks.py --mode distributed --batch_size $BATCH_SIZE_DISTRIBUTED --models 'llama' --precision 'float32' --sdc_check 1 --random_seed 1 --duration $DURATION --num_steps 1000000 | tee run_distributed.log
  python run_benchmarks.py --mode concurrent --batch_size $BATCH_SIZE_CONCURRENT --models 'llama' --precision 'float32' --sdc_check 1 --random_seed 1 --duration $DURATION --num_steps 1000000 | tee run_concurrent.log
  python analytics_gpu.py --distributed_target_step_time $DISTRIBUTED_TARGET_STEP_TIME --concurrent_target_step_time $CONCURRENT_TARGET_STEP_TIME --percentage_threshold 3 --ref_log_file ./ref_log/ref_gtt_h100_80gb_dist.txt | tee "$CWD"/analytics_gpudiag.log
# 40G A100 memory.toal 42949672960
elif jq -e '(.gpu_info.gpu_details[0].detail.nvidia_info.memory.total == 42949672960)' <<< "$gpu_info" > /dev/null; then
  BATCH_SIZE_DISTRIBUTED=5
  BATCH_SIZE_CONCURRENT=7
  DISTRIBUTED_TARGET_STEP_TIME=380
  CONCURRENT_TARGET_STEP_TIME=372
  echo "Found 40G A100, set distributed batch size to $BATCH_SIZE_DISTRIBUTED, and concurrent batch size to $BATCH_SIZE_CONCURRENT"
  python run_benchmarks.py --mode distributed --batch_size $BATCH_SIZE_DISTRIBUTED --models 'llama' --precision 'float32' --sdc_check 1 --random_seed 1 --duration $DURATION --num_steps 1000000 | tee run_distributed.log
  python run_benchmarks.py --mode concurrent --batch_size $BATCH_SIZE_CONCURRENT --models 'llama' --precision 'float32' --sdc_check 1 --random_seed 1 --duration $DURATION --num_steps 1000000 | tee run_concurrent.log
  python analytics_gpu.py --distributed_target_step_time $DISTRIBUTED_TARGET_STEP_TIME --concurrent_target_step_time $CONCURRENT_TARGET_STEP_TIME --percentage_threshold 3 --ref_log_file ./ref_log/ref_zion_a100_40gb_dist.txt | tee "$CWD"/analytics_gpudiag.log

else
  echo "GPU not recognized"
  exit 1
fi


cp run_distributed.log "$CWD"/run_distributed.log
cp run_concurrent.log "$CWD"/run_concurrent.log
