# __Deep Learning Benchmarking Suite__
Deep Learning Benchmarking Suite (DLBS) is a collection of tools for providing consistent and reproducible benchmark experiments on various hardware/software combinations. In particular, DLBS provides the following functionality:
1. Implements internally various deep models. Our goal is to provide same model implementations for all supported frameworks. Deep models that are supported include various VGGs, ResNets, AlexNet, GoogleNet and others.
2. Benchmarks single node CPU/multi-GPU configurations. Frameworks that are now supported: BVLC/NVIDIA/Intel Caffe, Caffe2, TensorFlow, MXNet and TensorRT. Due to rapid development progress of these frameworks, we fix framework versions to particular commit that we have tested.
3. Supports inference and training phases.
4. Benchmarking tools can use real data if dataset is available. Else, falls back to synthetic data.
5. Supports bare metal and docker environments.

## Supported platforms
Deep Learning Benchmarking Suite was tested on various servers with Ubuntu /
RedHat / CentOS operating systems with/without NVIDIA GPUs. It may not work with
Mac OS due to slightly different command line API of some of the tools we use
(like, for instance, sed) - we will fix this in one of the next releases.

## Installation
1. Install Docker and NVIDIA Docker for containerized benchmarks. Read [here](/docker/docker.md?id=docker) why we prefer to use docker and [here](/docker/install_docker.md?id=installing-docker) for installing/troubleshooting tips. This is not required. DLBS can work with bare metal framework installations.
2. Clone Deep Learning Benchmarking Suite from [GitHub](https://github.com/HewlettPackard/dlcookbook-dlbs.git)
   ```bash
   git clone https://github.com/HewlettPackard/dlcookbook-dlbs dlbs
   ```
3. Build/pull docker images for containerized benchmarks or build/install host frameworks for bare metal benchmarks.
    1. [TensorFlow](http://tensorflow.org)
    2. [BVLC Caffe](http://caffe.berkeleyvision.org/)
    3. [NVIDIA Caffe](https://github.com/NVIDIA/caffe)
    4. [Intel Caffe](https://github.com/intel/caffe)
    5. [Caffe2](http://caffe2.ai)
    6. [MXNet](http://mxnet.io)
    7. [TensorRT](https://developer.nvidia.com/tensorrt)

   There are several ways to get Docker images. Read [here](/docker/pull_build_images.md?id=buildpull-docker-images) about various options.

## Quick start
Assuming TensorFlow is installed and CUDA enabled GPU is present, execute the following commands to run simple experiment with ResNet50 model (if you do not have GPUs, see below):
```bash
# Go to DLBS home folder
cd dlbs
# Build TensorFlow image that's set as default in standard configuration files.
# Alternatively, you can skip this step and use your own image or pull image from NVIDIA GPU Cloud
cd ./docker
./build tensorflow/cuda9-cudnn7
cd ..
# Setup python paths
export PYTHONPATH=$(pwd)/python:$PYTHONPATH
# Run experiment. It will run containerized GPU TensorFlow with default image 'hpe/tensorflow:cuda9-cudnn7'
# If you want to use your own image, add this argument: -Ptensorflow.docker_image='"YOUR_DOCKER_IMAGE_NAME"'
python ./python/dlbs/experimenter.py run -Pexp.framework='"tensorflow"' -Pexp.model='"resnet50"' -Pexp.gpus='"0"' -Pexp.bench_root='"./benchmarks/my_experiment"' -Pexp.log_file='"./benchmarks/my_experiment/tf.log"'
# Print some results
python ./python/dlbs/logparser.py --keys exp.device_type results.time exp.framework_title exp.model_title exp.replica_batch -- ./benchmarks/my_experiment/tf.log
```

If you do not have NVIDIA GPUs, run TensorFlow in CPU mode (the only difference is that
GPUs set to empty string: `--exp.gpus=""`):
```bash
# First steps same as in above GPU example - go to DLBS root folder and build/pull image.
# You may want to build a CPU only version of TensorFlow. By default, experimenter will use
# 'docker' to run CPU workloads what may not work. In the example below I override this
# behavior by providing exp.docker_launcher parameter.
cd dlbs
# Setup python paths
export PYTHONPATH=$(pwd)/python:$PYTHONPATH
# Run experiment
python ./python/dlbs/experimenter.py run -Pexp.framework='"tensorflow"' -Pexp.model='"resnet50"' -Pexp.gpus='""' -Pexp.log_file='"./benchmarks/my_experiment/tf.log"' -Pexp.docker_launcher='"nvidia-docker"'
# Print some results
python ./python/dlbs/logparser.py --keys exp.device_type results.time exp.framework_title exp.model_title exp.replica_batch -- ./benchmarks/my_experiment/tf.log
```

If everything is OK, you should expect seeing this JSON (training time - an average batch time - of course will be different depending on your GPU/CPU models):
```json
{
    "data": [
        {
            "exp.device_type": "gpu",
            "exp.replica_batch": "16",
            "exp.framework_title": "TensorFlow",
            "exp.model_title": "ResNet50",
            "results.time": 255.59105431309905
        }
    ]
}
```

If `results.time` is not there, study ./benchmarks/my_experiment/tf.log for error messages.



## Deep Learning CookBook
Deep Learning Benchmarking Suite is part of HPE's Deep Learning CookBook project.
A project overview can be found on HPE developer portal [here](https://developer.hpe.com/platform/deep-learning-cookbook/home)

## Documentation

We host documentation on GitHub pages [here](http://hewlettpackard.github.io/dlcookbook-dlbs).

## License

Deep Learning Benchmarking Suite is released under the [Apache 2.0 license](./LICENSE).

## Contact us

* Natalia Vassilieva <nvassilieva@hpe.com>
* Sergey Serebryakov <sergey.serebryakov@hpe.com>
