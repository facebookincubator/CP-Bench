#!/bin/bash
export NCCL_DEBUG=0
# Get the current working directory
CWD=$PWD
cd "$(dirname "$0")"
# Default values
DURATION=300         # 5 minutes
BATCH_SIZE=20480
NPROC_on_node=2
# Parse arguments
while getopts ":t:b:n:" opt; do
  case $opt in
    t) DURATION="$OPTARG";;      # -t for duration
    b) BATCH_SIZE="$OPTARG";;    # -b for batch size
    n) NPROC_on_node="$OPTARG";; # -n for number of processes
    \?) echo "Invalid option -$OPTARG"; exit 1;;
  esac
done

echo "***start running DLRM distributed training for $DURATION seconds***"

cd ./dlrm-main

torchrun --nproc_per_node=$NPROC_on_node dlrm_s_pytorch.py \
    --arch-embedding-size="80000-80000-80000-80000-80000-80000-80000-80000" \
    --arch-sparse-feature-size=128 \
    --arch-mlp-bot="128-128-128-128" \
    --arch-mlp-top="512-51200-51200-25600-1" \
    --max-ind-range=40000000 \
    --data-generation=random \
    --loss-function=bce \
    --round-targets=True \
    --learning-rate=0.1 \
    --mini-batch-size=$BATCH_SIZE \
    --print-freq=2 \
    --print-time \
    --test-freq=2 \
    --test-mini-batch-size=2048 \
    --memory-map \
    --use-gpu \
    --num-batches=100 \
    --dist-backend=nccl \
    --nepochs=100000 \
    --duration=$DURATION

echo "done training for $DURATION seconds"
