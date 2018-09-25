# Multi-GPU compute scaling test

#### Introduction
The purpose of this recipe is to quickly test if your GPUs and GPU to GPU
communication links (PCIe/NVLINK) are properly configured and are capable of
running multi-GPU training workloads as expected demonstrating good scaling.
There is no input data set - synthetic data is used. This workload tests GPUs
and GPU to GPU communication.

#### Prerequisites
This workload depends on NGC container `nvcr.io/nvidia/tensorflow:18.04-py3`. To get
it, run the following (you need to be registered at [NGC](http://ngc.nvidia.com)):
```bash
docker login nvcr.io
docker pull nvcr.io/nvidia/tensorflow:18.04-py3
docker logout nvcr.io
```

#### Recipe description
The workload is a single/multi-GPU ResNet50 training. Replica batch size is 128,
weak scaling strategy is utilized as number of GPUs increases, half(mixed)
precision is used. Benchmark configuration is specified in [config.json](./config.json)
file. To run it, launch [run](./run) bash script.

In default configuration, number of warmup and benchmark iterations is small and
thus performance results should not be used for performance analysis. You want to
increase number of warmup iterations to 100 and number of benchmark iterations to
~ 400 to get more reliable results:
```json
{
  "exp.num_warmup_batches": 100,
  "exp.num_batches": 400,
}
```

If your node has less than 8 GPUs, adjust `exp.gpus` parameter accordingly.

### Expected results
Results will be saved to `reports` folder. You should normally find there
`results.json` file with benchmark summary, `results.txt` file with weak scaling
reports and `results.png` with scaling chart if matplotlib is installed. If you
observe error messages or there are no files in `reports` folder or report files
are empty, study log files in `logs` directory.

ResNet50 model should scale almost linearly with both SXM2 and PCIe GPUs. For
reference numbers, consult
[HPE Deep Learning Performance Guide](https://dlpg.labs.hpe.com).
