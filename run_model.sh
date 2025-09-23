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


while getopts ":g:t:" opt; do
  case $opt in
    g) GPU_TYPE="$OPTARG";;
    t) DURATION="$OPTARG";;
    \?) echo "Invalid option: -$OPTARG"; exit 1;;
  esac
done

# IMPORTANT: please obtain your own reference log file here based on your environment by running cp-bench on a healthy machine first
case $GPU_TYPE in
  h100_96gb)
    NPROC_PER_NODE=8
    END_GPU=7
    BATCH_SIZE_DISTRIBUTED=32
    BATCH_SIZE_CONCURRENT=36
    DISTRIBUTED_TARGET_STEP_TIME=735
    CONCURRENT_TARGET_STEP_TIME=749
    REF_LOG_FILE=./ref_log/ref_gtt_h100_96gb_dist.txt
    ;;
  a100_80gb)
    NPROC_PER_NODE=8
    END_GPU=7
    BATCH_SIZE_DISTRIBUTED=24
    BATCH_SIZE_CONCURRENT=28
    DISTRIBUTED_TARGET_STEP_TIME=1021
    CONCURRENT_TARGET_STEP_TIME=1110
    REF_LOG_FILE=./ref_log/ref_zion_a100_80gb_dist.txt
    ;;
  h100_80gb)
    NPROC_PER_NODE=8
    END_GPU=7
    BATCH_SIZE_DISTRIBUTED=24
    BATCH_SIZE_CONCURRENT=28
    DISTRIBUTED_TARGET_STEP_TIME=477
    CONCURRENT_TARGET_STEP_TIME=513
    REF_LOG_FILE=./ref_log/ref_gtt_h100_80gb_dist.txt
    ;;
  a100_40gb)
    NPROC_PER_NODE=8
    END_GPU=7
    BATCH_SIZE_DISTRIBUTED=5
    BATCH_SIZE_CONCURRENT=7
    DISTRIBUTED_TARGET_STEP_TIME=380
    CONCURRENT_TARGET_STEP_TIME=372
    REF_LOG_FILE=./ref_log/ref_zion_a100_40gb_dist.txt
    ;;
  gb200_186gb)
    NPROC_PER_NODE=2
    END_GPU=1
    BATCH_SIZE_DISTRIBUTED=64
    BATCH_SIZE_CONCURRENT=64
    DISTRIBUTED_TARGET_STEP_TIME=545
    CONCURRENT_TARGET_STEP_TIME=525
    REF_LOG_FILE=./ref_log/ref_catalina_gb200_186gb_dist.txt
    ;;
  *)
    echo "Invalid GPU type"
    exit 1
    ;;
esac
echo "Found $GPU_TYPE, set distributed batch size to $BATCH_SIZE_DISTRIBUTED, and concurrent batch size to $BATCH_SIZE_CONCURRENT"
python run_benchmarks.py --mode distributed --batch_size $BATCH_SIZE_DISTRIBUTED --models 'llama' --precision 'float32' --sdc_check 1 --random_seed 1 --duration $DURATION --num_steps 1000000 --nproc_per_node $NPROC_PER_NODE | tee run_distributed.log
python run_benchmarks.py --mode concurrent --batch_size $BATCH_SIZE_CONCURRENT --models 'llama' --precision 'float32' --sdc_check 1 --random_seed 1 --duration $DURATION --num_steps 1000000 --end-gpu $END_GPU | tee run_concurrent.log
python analytics_gpu.py --distributed_target_step_time $DISTRIBUTED_TARGET_STEP_TIME --concurrent_target_step_time $CONCURRENT_TARGET_STEP_TIME --percentage_threshold 3 --ref_log_file $REF_LOG_FILE | tee "$CWD"/analytics_gpu.log

cp run_distributed.log "$CWD"/run_distributed.log
cp run_concurrent.log "$CWD"/run_concurrent.log
