# Single process inference benchmarks

Single-process multi-GPU inference benchmark is just a synonym term for regular benchmarks. One benchmark process
is run that can use multiple GPUs to run inference. 

We recommend running it with DLBS that can take care about passing right parameters and running benchmarks in docker
containers. The framework name for it is `tensorrt` and it uses standard parameter names as for any other frameworks
i.e. `exp.gpus`, `exp.model`, `exp.data_dir` etc.

## Command line parameters
TensorRT benchmark backend is configured with two types of parameters - those provided on a command line and those
provided via environment.

### Common parameters

1. `--gpus` A comma seperated list of GPU identifiers to use.
2. `--model` Model identifier like alexnet, resent18 etc. Used to store calibration caches.
3. `--model_file` Caffe's prototxt deploy (inference) model. Because this benchmark backend is tightly integrated with DLBS,
   prototxt files loaded by the tool may be slightly different from standard Caffe's descriptors.
4. `--batch_size` Per device batch size. In case of multiple GPUs, weak scaling strategy is used.
5. `--dtype` Type of data variables: float(same as float32), float32, float16 or int8.
6. `--num_warmup_batches` Number of warmup iterations.
7. `--num_batches` Number of benchmark iterations.
8. `--inference_queue_size` Number of pre-allocated inference requests. This value can be identified experimentally. Initial
   guess can be numebr of GPUs * 3.
9. `--cache` Path to folder that will be used to store models calibration data and serialzied engines between benchmarks.

### Parameters related to input data

1. `--data_dir` Path to a dataset.
2. `--data_name` Name of a dataset - 'images', 'tensors1' or 'tensors4'.

#### Common parameters for all non-synthetic datasets:
1. `--num_prefetchers` Number of prefetch threads (data readers).

#### Parameters for custom dataset (tensors1 and tensors4)
There are no specific parametes except those configured via environment variables (see below). One thing to note is that
when using this dataset, prefetchers (data readers) read images and batch them together using the same batch size used by
inference engine.

#### Parameters for `images` dataset (raw JPEG files):
This is not recommented way to run benchmarks. It's not optimzied and performance will be unsatisfactory due to slow
data ingestion pipeline - data pipeline will always be a bottleneck.

As opposed to `tensors` dataset, this dataset internally stages two operations - reading data and preprocessing it.

1. `--resize_method` How to resize images: 'crop' or 'resize'.
2. `--prefetch_queue_size` Internal queue used to communicate images between prefetchers and decoders. This is maximal
   size if the queue.
3. `--prefetch_batch_size` Size of a prefetch batch which can be different from inference batch. Prefetchers use this
   size to batch images together and send them via prefetch queue to decoders.
4. `--num_decoders` Number of decoder threads (that convert JPEG to input blobs). Decoders also make sure that their
   outputs contains correct number of images.

### Parameters that most likely should not be changed when running with DLBS

1. `--input` Name of an input data tensor (data)."
2. `--output` Name of an output data tensor (prob).

### Additional parameters

1. `--profile` Profile model and report results.
2. `--report_frequency` Report performance every 'report_frequency' processed batches. Default (-1) means report in the end.
   For benchmarks that last not very long time this may be a good option. For very long lasting benchmarks, set this to some
   positive value.
3. `--no_batch_times` Do not collect and report individual batch times. You may want not to report individual batch times when
   running very long lasting benchmarks. Usually, it's used in combination with --report_frequency=N. If you do not set the 
   report_frequency and use no_batch_times, the app will still be collecting batch times but will not log them.


## Environment parameters
Search for `Environment` class and read its description. That description provides overview of all environment parameters that
affect performance of this benchmark tool.

### Running benchmarks with DLBS
TensorRT benchmark backend is natively integrated with DLBS. This means that it supports the standard set of parameters such
as `exp.gpus`, `exp.model` etc. As usually, it is convinient to run benchmarks providing configuration in an input JSON file.
For example, we used the following configuration to run storage benchmarks:
```json
{
    "parameters": {
        "exp.num_warmup_batches": 100,
        "exp.num_batches": 400,
        "monitor.frequency": 0,
        "exp.status": "disabled",
        "exp.log_file": "${BENCH_ROOT}/logs/$(\"${exp.gpus}\".replace(\",\",\".\"))$_${exp.model}_${exp.effective_batch}.log",
        "exp.docker": true,
        "tensorrt.docker_image": "dlbs/tensorrt:18.10",
        "exp.framework": "tensorrt",
        "exp.phase": "inference",
        "exp.dtype": "float16",

        "tensorrt.data_dir": "/some/hpc/storage/imagenet100k",
        "tensorrt.data_name": "tensors1",
        "exp.data_store": "Human readeable description of the storage goes here",

        "tensorrt.num_prefetchers": "16",
        "tensorrt.inference_queue_size": "96",

        "exp.docker_args": "--rm --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864"
    },
    "variables": {
        "exp.gpus": ["0","0,4","0,2,4,6","0,1,2,3,4,5,6,7"],
        "exp.model": ["alexnet_owt", "resnet152", "resnet50"],
        "exp.replica_batch": [64, 128, 256, 512, 1024]
    },
    "extensions": [
        {
            "condition": {"exp.model": "alexnet_owt", "exp.replica_batch": [512, 1024]},
            "parameters": {"exp.status":"", "exp.num_batches": 600}
        },
        {
            "condition": {"exp.model": "resnet152", "exp.replica_batch": [64,128]},
            "parameters": {"exp.status":"", "exp.num_batches": 350}
        },
        {
            "condition": {"exp.model": "resnet50", "exp.replica_batch": [128, 256]},
            "parameters": {"exp.status":"", "exp.num_batches": 500}
        }
    ]
}

```
Assuming that this json file is located, for instance, in DLBS_ROOT/benchmarks/storage/step05/sprocess, the following
bash script runs this configuration:
```bash
#!/bin/bash

export BENCH_ROOT=$( cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )
export CUDA_CACHE_PATH=/dev/shm/cuda_cache
. ${BENCH_ROOT}/../../../../scripts/environment.sh

script=$DLBS_ROOT/python/dlbs/experimenter.py            # Script that runs benchmarks
parser=$DLBS_ROOT/python/dlbs/logparser.py               # Script that parses log files
sb=$DLBS_ROOT/python/dlbs/reports/summary_builder.py     # Script that builds reports

# Run experiments.
/usr/bin/python $script run --log-level=info --config=./config.json

# Parse log files
[ -f ./results.json ] && rm ./results.json
params="exp.status,exp.framework_title,exp.effective_batch,results.time,results.throughput,exp.model_title,exp.gpus"
/usr/bin/python $parser ./logs/*.log --output_params ${params} --output_file ./results.json

# Build weak scaling report
/usr/bin/python $sb --summary_file ./results.json --type weak-scaling --target_variable results.time > ./results.txt

```