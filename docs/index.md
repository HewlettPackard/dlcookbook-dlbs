# __Deep Learning Benchmarking Suite__
Deep Learning Benchmarking Suite (DLBS) is a collection of command line tools for providing consistent and reproducible benchmark experiments on various hardware/software combinations. In particular, DLBS provides the following functionality:
1. Implements internally various deep models. Our goal is to provide same model implementations for all supported frameworks. Deep models that are supported include various VGGs, ResNets, AlexNet and GoogleNet models.
2. Benchmarks single node multi GPU configurations. Frameworks that are now supported: BVLC Caffe, NVIDIA Caffe, Intel Caffe, Caffe2, TensorFlow, MXNet, PyTorch and NVIDIA inference engine TensorRT.
3. Supports inference and training phases.
4. Can use real data if dataset is available. Else, falls back to synthetic data.
5. Supports bare metal and docker environments.
6. Can run benchmarks with single, half and in some cases int8 precision.

## Supported platforms
Deep Learning Benchmarking Suite was tested on various servers with Ubuntu /
RedHat / CentOS operating systems with/without NVIDIA GPUs. It may not work with
Mac OS due to slightly different command line API of some of the tools we use
(like, for instance, sed) - we will fix this in one of the next releases.

## Installation
1. Install Docker and NVIDIA Docker for containerized benchmarks. Read [here](/docker/docker.md?id=docker) why we prefer to use docker and [here](/docker/install_docker.md?id=installing-docker) for installing/troubleshooting tips. This is not required. DLBS can work with bare metal framework installations.
2. Clone Deep Learning Benchmarking Suite from [GitHub](https://github.com/HewlettPackard/dlcookbook-dlbs)
   ```bash
   git clone https://github.com/HewlettPackard/dlcookbook-dlbs dlbs
   ```
3. The benchmarking suite mostly uses modules from standard python library (python 2.7). Optional dependencies that do not influence the benchmarking process are listed in `python/requirements.txt`. If they are not found, the code that uses it will be disabled.
4. Build/pull docker images for containerized benchmarks or build/install host frameworks for bare metal benchmarks.
    1. [TensorFlow](http://tensorflow.org)
    2. [BVLC Caffe](http://caffe.berkeleyvision.org/)
    3. [NVIDIA Caffe](https://github.com/NVIDIA/caffe)
    4. [Intel Caffe](https://github.com/intel/caffe)
    5. [Caffe2](http://caffe2.ai)
    6. [MXNet](http://mxnet.io)
    7. [TensorRT](https://developer.nvidia.com/tensorrt)
    8. [PyTorch](http://pytorch.org/)

   There are several ways to get Docker images. Read [here](/docker/pull_build_images.md?id=buildpull-docker-images) about various options including images from [NVIDIA GPU Cloud](https://www.nvidia.com/en-us/gpu-cloud/). We may not support the newest framework versions due to API change.

## Quick start
Assuming TensorFlow is installed and CUDA enabled GPU is present, execute the following commands to run simple experiment with ResNet50 model  (if you do not have GPUs, see below):
```bash
# Go to DLBS home folder
cd dlbs
# Setup python paths
export PYTHONPATH=$(pwd)/python:$PYTHONPATH
# Create folder for experiment results
mkdir -p ./benchmarks/my_experiment
# Run experiment
python ./python/dlbs/experimenter.py run -Pexp.framework='"tensorflow"' -Pexp.model='"resnet50"' -Pexp.gpus='"0"' -Pexp.log_file='"./benchmarks/my_experiment/tf.log"'
# Print some results
python ./python/dlbs/logparser.py ./benchmarks/my_experiment/tf.log --output_params "exp.device_type,exp.phase,results.time,exp.framework_title,exp.model_title,exp.replica_batch,exp.framework_ver"
```

If you do not have NVIDIA GPUs, run TensorFlow in CPU mode (the only difference is that
GPUs set to empty string: `--exp.gpus=""`):
```bash
# Go to DLBS home folder
cd dlbs
# Setup python paths
export PYTHONPATH=$(pwd)/python:$PYTHONPATH
# Create folder for experiment results
mkdir -p ./benchmarks/my_experiment
# Run experiment
python ./python/dlbs/experimenter.py run -Pexp.framework='"tensorflow"' -Pexp.model='"resnet50"' -Pexp.device_type='"cpu"' -Pexp.log_file='"./benchmarks/my_experiment/tf.log"'
# Print some results
python ./python/dlbs/logparser.py ./benchmarks/my_experiment/tf.log --output_params "exp.device_type,exp.phase,results.time,exp.framework_title,exp.model_title,exp.replica_batch,exp.framework_ver"
```

If everything is OK, you should expect seeing this JSON (training time - an average batch time - of course will be different):
```json
{
    "data": [
        {
            "exp.device_type": "gpu",
            "exp.framework_title": "TensorFlow",
            "exp.framework_ver": "1.4.0",
            "exp.model_title": "ResNet50",
            "exp.phase": "training",
            "exp.replica_batch": 16,
            "results.time": 273.27070879590093
        }
    ]
}
```

If `results.time` is not there, study ./benchmarks/my_experiment/tf.log for error messages.

## Further reading

- [Docker](/docker/docker.md?id=docker)
  - [Install](/docker/install_docker.md?id=installing-docker)
  - [Network](/docker/docker_network.md?id=docker-networking)
  - [Pull/build images](/docker/pull_build_images.md?id=buildpull-docker-images)
- [Introduction](/intro/intro.md?id=introduction)
- [Tutorials](/tutorials/tutorials.md?id=tutorials)
- [Frameworks](/frameworks/frameworks.md?id=frameworks)
  - [Caffe](/frameworks/caffe.md?id=caffe)
  - [Caffe2](/frameworks/caffe2.md?id=caffe2)
  - [MXNet](/frameworks/mxnet.md?id=mxnet)
  - [TensorFlow](/frameworks/tensorflow.md?id=tensorflow)
  - [TensorRT](/frameworks/tensorrt.md?id=tensorrt)
  - [PyTorch](/frameworks/pytorch.md?id=pytorch)
- [Extending DLBS](/extend/dlbs.md?id=extending-deep-learning-benchmarking-suite)
- [Resource monitor](/monitor/monitor.md?id=resource-monitor)

## Contact us

* Natalia Vassilieva <nvassilieva@hpe.com>
* Sergey Serebryakov <sergey.serebryakov@hpe.com>
