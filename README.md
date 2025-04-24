# CP-Bench: Configurable and Paramraterizable PyTorch Testing to Detect GPU Silent Data Corruption, Measure Perf, and Stress Testing

CP-Bench, stands for Configurable and Paramraterizable PyTorch-level Test Benchmark, is an open-source test tool designed to detect GPU silent data corruption, measure GPU perf, and stress test GPUs.

## Features

* **GPU Silent Data Corruption Detection**: CP-Bench uses **PyTorch's deterministic running** to detect silent data corruption on GPUs, which can cause incorrect results without raising any hardware alerts.
* **Different Running Modes**: CP-Bench supports distributed mode, as well as concurrent modes, to support SDC check across hosts and GPUs.
* **Stress Testing**: The benchmark includes adaptive PyTorch workloads to stress test GPUs, identifying potential performance bottlenecks and stability issues.
* **Measure Perf**: The benchmark also measures GPU performance.
* **Supported Models**: LLAMA, GPT, BERT, LSTM, GEMM, ResNet



## Requirements

* **PyTorch**: CP-Bench requires PyTorch 1.7 or later to be installed.
* **CUDA**: A CUDA-compatible NVIDIA GPU is required for running the benchmark.
* **Python**: Python 3.6 or later is required to run the benchmark.
* **Additional Libraries**: torchvision, torchaudio, transformers, transformer_engine


## Prepare Conda Environment

To prepare Conda Env to run CP-Bench, follow these steps:

1. Install Conda first if you haven't already and Activate it.
```
# first install conda, then activate it
conda create -yn [your-env-name]
conda activate [your-env-name]
```

2. Install PyTorch and required libraries using conda:
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

3. Install additional libraries using pip:
```
pip install transformers
pip install transformer_engine
```

4. Clone this repository and navigate to the project directory:
```
git clone https://github.com/facebookincubator/CP-Bench.git
cd CP-Bench
```

## Running CP-Bench

1. You first need to generate a reference log file for the host/GPU you want to compare against. Because the checksum value could change with different envinroments, it is very important to run CP-Bench on a healthy host first to get a log file.

For example, if you are checking H100_96GB GPU, then you can run
```
python run_benchmarks.py --mode distributed --batch_size 32 --models 'llama' --precision 'float32' --sdc_check 1 --random_seed 1 --duration 7200 --num_steps 1000000 | tee ref_h100_96gb_dist.txt.log
```
This will serve as your reference log file for H100_96GB GPU.
Note, you might need to adjust parameters such as batch_size to avoid CUDA out of memory errors for different GPU types.

You can find other examples in the run_model.sh script to generate reference log files for other GPU types.

2. After you generating a reference log file, you can execute the run_model.sh script (note this scripts currently supports 4 GPU types, make adjustments if needed).

Very important: You need to modify the run_model.sh script to specify the reference log file you generated in step 1.

```
conda activate [your-env-name]
./run_model.sh -t 1800 -g h100_96GB
```
You can adjust the runtime of run_model.sh by providing -t option, and gpu type using -g option, e.g.,
```
# this will run  hour for distributed mode, and 1 hour for concurrent mode, for h100_80gb GPU
./run_model.sh -t 3600 -g h100_80gb
```

3. For more customized usage, you can modify the run_model.sh script to customize the benchmark parameters.

Or, you can run the benchmark manually by executing the following commands:
```
# Concurrent mode: Each GPU run independently
export CUBLAS_WORKSPACE_CONFIG=:4096:8 && python run_benchmarks.py --mode concurrent --batch_size 28 --models "llama" --precision "float32" --sdc_check 1 --random_seed 1 --duration 10000 --num_steps 1000000 |tee run_concurrent.log
```

```
# Distributed mode: All GPUs run together for one model.
export NCCL_DEBUG=0 && export CUBLAS_WORKSPACE_CONFIG=:4096:8 && python run_benchmarks.py --mode distributed --batch_size 24 --models "llama" --precision "float32" --sdc_check 1 --random_seed 1 --duration 10000 --num_steps 1000000 |tee run_distributed_llama.log
```

## Real-World Use Cases
We have used CP-Bench detecting real-world SDCs and Perf Throttling of GPUs.

Example on Silent Data Corruption:
```
# After running ./run_model.sh on a host, the last lines of the script outputs the following:
*************************************************************************************************
*************************************************************************************************
******************************** Perform Testing Analysis ***************************************
*************************************************************************************************
*************************************************************************************************
PASS: Distributed Stress Testing.
PASS: Concurrent stress testing.
PASS (distributed perf): Actual perf 482.8630452804772, target perf 477.0, diff <= threshold 3.0%
PASS (individual_gpu perf): Actual perf 519.3224020060536, target perf 513.0, diff <= threshold 3.0% for GPU device 2.
PASS (individual_gpu perf): Actual perf 521.0784714482176, target perf 513.0, diff <= threshold 3.0% for GPU device 4.
PASS (individual_gpu perf): Actual perf 514.4733273197381, target perf 513.0, diff <= threshold 3.0% for GPU device 3.
PASS (individual_gpu perf): Actual perf 514.6490608040447, target perf 513.0, diff <= threshold 3.0% for GPU device 5.
PASS (individual_gpu perf): Actual perf 515.775753297232, target perf 513.0, diff <= threshold 3.0% for GPU device 7.
PASS (individual_gpu perf): Actual perf 516.6577360349945, target perf 513.0, diff <= threshold 3.0% for GPU device 0.
PASS (individual_gpu perf): Actual perf 517.0651094278155, target perf 513.0, diff <= threshold 3.0% for GPU device 6.
PASS (individual_gpu perf): Actual perf 517.6505637373829, target perf 513.0, diff <= threshold 3.0% for GPU device 1.
FAIL: Distributed SDC testing at step 200.
Step 200: Checksum values are inconsistent across GPU ranks.
FAIL: Concurrent SDC testing.
```

This shows that stress tests and perf tests passed, but SDC is detected on both distributed mode run and concurrent mode run.
Specifically, concurrent run fails at step 200, indicated by the inconsistency across individual GPUs.
So checking the log, we found at step 200, GPU 0 is producing a different checksum value than all other GPUs.
That is where we identify GPU 0 is having SDC.
```
[2025-03-20 20:00:31,356 ][pytorch_llama.py:235][INFO] Checksum at step 100: 87852.31676383689 at GPU rank 0
[2025-03-20 20:00:32,121 ][pytorch_llama.py:235][INFO] Checksum at step 100: 87852.31676383689 at GPU rank 3
[2025-03-20 20:00:32,162 ][pytorch_llama.py:235][INFO] Checksum at step 100: 87852.31676383689 at GPU rank 5
[2025-03-20 20:00:32,336 ][pytorch_llama.py:235][INFO] Checksum at step 100: 87852.31676383689 at GPU rank 7
[2025-03-20 20:00:32,371 ][pytorch_llama.py:235][INFO] Checksum at step 100: 87852.31676383689 at GPU rank 6
[2025-03-20 20:00:32,491 ][pytorch_llama.py:235][INFO] Checksum at step 100: 87852.31676383689 at GPU rank 1
[2025-03-20 20:00:32,560 ][pytorch_llama.py:235][INFO] Checksum at step 100: 87852.31676383689 at GPU rank 2
[2025-03-20 20:00:32,895 ][pytorch_llama.py:235][INFO] Checksum at step 100: 87852.31676383689 at GPU rank 4
[2025-03-20 20:01:23,551 ][pytorch_llama.py:235][INFO] Checksum at step 200: 87853.47495869175 at GPU rank 0
[2025-03-20 20:01:24,035 ][pytorch_llama.py:235][INFO] Checksum at step 200: 87853.47548914701 at GPU rank 3
[2025-03-20 20:01:24,085 ][pytorch_llama.py:235][INFO] Checksum at step 200: 87853.47548914701 at GPU rank 5
[2025-03-20 20:01:24,471 ][pytorch_llama.py:235][INFO] Checksum at step 200: 87853.47548914701 at GPU rank 7
[2025-03-20 20:01:24,566 ][pytorch_llama.py:235][INFO] Checksum at step 200: 87853.47548914701 at GPU rank 6
[2025-03-20 20:01:24,854 ][pytorch_llama.py:235][INFO] Checksum at step 200: 87853.47548914701 at GPU rank 1
[2025-03-20 20:01:25,052 ][pytorch_llama.py:235][INFO] Checksum at step 200: 87853.47548914701 at GPU rank 2
[2025-03-20 20:01:25,454 ][pytorch_llama.py:235][INFO] Checksum at step 200: 87853.47548914701 at GPU rank 4
```

Example on Perf Throttling:
This example shows that GPU 3 is hitting a perf throttling.
```
# After running ./run_model.sh on a host, the last lines of the script outputs the following:
***********************************************************************************************
***********************************************************************************************
****************************** Perform Testing Analysis *************************************
***********************************************************************************************
***********************************************************************************************
PASS: Distributed Stress Testing.
PASS: Concurrent stress testing.
WARNING (distributed perf): Actual perf 518.635, target perf 477.0, diff > threshold 3.0%
PASS (individual_gpu perf): Actual perf 519.681, target perf 513.0, diff <= threshold 3.0% for GPU device 4.
PASS (individual_gpu perf): Actual perf 519.722, target perf 513.0, diff <= threshold 3.0% for GPU device 2.
WARNING (individual_gpu perf): Actual perf 555.011, target perf 513.0, diff > threshold 3.0% for GPU device 3.
PASS (individual_gpu perf): Actual perf 513.092, target perf 513.0, diff <= threshold 3.0% for GPU device 5.
PASS (individual_gpu perf): Actual perf 515.608, target perf 513.0, diff <= threshold 3.0% for GPU device 1.
PASS (individual_gpu perf): Actual perf 516.146, target perf 513.0, diff <= threshold 3.0% for GPU device 0.
PASS (individual_gpu perf): Actual perf 516.396, target perf 513.0, diff <= threshold 3.0% for GPU device 7.
PASS (individual_gpu perf): Actual perf 516.191, target perf 513.0, diff <= threshold 3.0% for GPU device 6.
PASS: Distributed SDC testing.
PASS: Concurrent SDC testing.
```


## Contributing

We welcome contributions to CP-Bench! If you'd like to add new tests, fix existing issues, or improve the benchmark in any way, please fork this repository and submit a pull request.


## License

CP-Bench is released under the MIT License.

## Acknowledgments

We would like to thank the Microsoft team for their open-source project SuperBench that CP-Bench is built upon.
