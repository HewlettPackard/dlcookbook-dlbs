# OpenVINO benchmarks recipes.

> Work in progress.
For more details, read this [file](../../../docker/openvino/19.09/README.md).


## Model Zoo
OpenVINO backend does not accept regular model names as other backends. Instead, models are specified how they are listed in OpenVINO model zoo. So far, only models in OpenVINO format (*.xml, *.bin) format were tested.

- Container: __dlbs/openvino:19.09__. OpenVINO: 2019.3.334. Model Zoo: [here](https://download.01.org/opencv/2019/open_model_zoo/R3/20190905_163000_models_bin/). 

## Model specification
As in any other benchmark backend, models are specified using `exp.model` parameter. By default, OpenVINO backend does not provide model names, so, users are responsible for providing names. In simplest case, the following parameter in a JSON configuration file will work:
```json
{
  "parameters": {
    "exp.model_title": "${exp.model}"
  }
}
```

Model name consists of a model name followed by a model selector, of instance, consider the following model: [resnet-50-int8-tf-0001](https://download.01.org/opencv/2019/open_model_zoo/R3/20190905_163000_models_bin/resnet-50-int8-tf-0001/). Tow possible models can be used with DLBS:
- "exp.model": "resnet-50-int8-tf-0001/FP16"
- "exp.model": "resnet-50-int8-tf-0001/FP32" 

where FP16 and FP32 are called selectors here.

## Performance optimization
Depending on your system, you may want to limit number of threads and pin benchmark app to specific cores. Imagine, that a system has two CPUs with hyper-threading:
```
NUMA node0 CPU(s):   0-19,40-59
NUMA node1 CPU(s):   20-39,60-79
```
Following options are possible:
- `"runtime.launcher": "OMP_NUM_THREADS=20 numactl --localalloc --physcpubind=0-19"` runs OpenVINO benchmarks on a CPU 0 with local memory allocation policy without hyper-threading.
- `"runtime.launcher": "OMP_NUM_THREADS=40 numactl --localalloc --physcpubind=0-19,40-59"` runs OpenVINO benchmarks on a CPU 0 with hyper-threading.

This works in a straightforward way. The OpenVINO bash launcher script runs the OpenVINO benchmarks similar to:
```bash
${runtime_launcher} ./benchmark_app ...
```
For better throughput, you probably need to use multiple instances of OpenVINO inference engine with parameters similar to those presented above. It will probably work better than increasing batch size.