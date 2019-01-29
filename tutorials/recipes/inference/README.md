# Baseline recipes to run inference benchmarks with synthetic data

This recipe uses one of the latest reference docker images `dlbs/tensorrt:18.12` that is based on NGC's `nvcr.io/nvidia/tensorrt:18.11-py3` image and does not require manual downloading of TensorRT package. It will however work with previous versions as well.

> This recipe was tested with Tesla P4/T4 GPUs with 1 and 2 GPUs in a single node. Though it can be used with larger number of GPUs, there's a better configuration for systems with 8 V100 or similar GPUs that will provide for certain non-compute intensive models better performance.

Build TensorRT docker image:
```bash
cd ${DLBS_ROOT}/docker
./build.sh tensorrt/18.12
```

If you need to use one of the previous docker images (for instance, because you already have it installed), change the following parameter:
```json
{"exp.docker_image": "dlbs/tensorrt:18.12"}
```

You may want to adjust [config](./config.json) file in this folder that defines benchmark configuration, in particular:

```json
{
  "exp.gpu_id": "tesla_t4",
  "exp.gpu_title": "Tesla T4"
}
```
The `exp.gpu_id` and `exp.gpu_title` parameters define your GPU model. These are human readable parameters and are usually used in a result analysis stage. Moreover, in the given configuration `exp.gpu_id` is used as a path element that defines locations of log files (see `exp.log_file`) parameter.

Probably, one of the most important sections is the section that defines what experiments are ran:
```json
"variables": {
    "exp.gpus": ["0"],
    "exp.model": ["alexnet_owt", "resnet152", "resnet50", "inception4", "vgg16", "googlenet", "vgg19", "resnet34", "resnet18", "acoustic_model", "deep_mnist"],
    "exp.replica_batch": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
    "exp.dtype": ["float32", "float16", "int8"]
}
```

This default configuration specifies the following:
1. Use 1 GPU with ID 0.
2. Use 11 neural networks.
3. Use 11 replica batch size.
4. Use three data types - single, half and int8.

This will result in `11*11*3 = 363` benchmarks in total (Cartesian product over these sets).

If your GPU does not support one of the data type, remove it from list.

In case you have multiple GPUs in your system, let's say 2, the following configuration will run benchmarks with 1 and on 2 GPUs:
```json
{"exp.gpus": ["0", "0,1"]}
```
With two and more GPUs, default distribution strategy is weak scaling (replica batch is a per-GPU batch size and never changes).
