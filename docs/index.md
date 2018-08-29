# __Deep Learning Benchmarking Suite__
Deep Learning Benchmarking Suite (DLBS) is a collection of command line tools for
running consistent and reproducible benchmark experiments on various hardware/software
platforms. In particular, DLBS:
1. Provides implementation of a number of neural networks in order to enforce apple-to-apple
   comparison across all supported frameworks. Deep models that are supported include various
   VGGs, ResNets, AlexNet and GoogleNet models. DLBS can support many more models via integration
   with third party benchmark projects.
2. Benchmarks single node multi-GPU or CPU  platforms. List of supported
   frameworks include various forks of Caffe (BVLC/NVIDIA/Intel), Caffe2, TensorFlow,
   MXNet, PyTorch. DLBS also supports NVIDIA's inference engine TensorRT.
3. Supports inference and training phases.
4. Supports synthetic and real data.
5. Supports bare metal and docker environments.
6. Supports single/half/int8 precision and uses tensor cores with Volta GPUs.
7. Is based on modular architecture enabling easy integration with other projects
   such TensorFlow CNN Benchmarks, Tensor2Tensor, NVCNN, NVCNN-HVD or similar.
8. Target metric is raw performance (number of data samples per second).

## Supported platforms
Deep Learning Benchmarking Suite was tested on various servers with Ubuntu /
RedHat / CentOS operating systems with and without NVIDIA GPUs. We have a little
success with running DLBS on top of AMD GPUs, but this is mostly untested. It may
not work with Mac OS due to slightly different command line API of some of the
tools we use (like, for instance, sed) - we will fix this in one of the next
releases.

## Installation
1. Install Docker and NVIDIA Docker for containerized benchmarks. Read [here](https://hewlettpackard.github.io/dlcookbook-dlbs/#/docker/docker?id=docker) why we prefer to use docker and [here](https://hewlettpackard.github.io/dlcookbook-dlbs/#/docker/install_docker?id=installing-docker) for installing/troubleshooting tips. This is not required. DLBS can work with bare metal framework installations.
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

   There are several ways to get Docker images. Read [here](https://hewlettpackard.github.io/dlcookbook-dlbs/#/docker/pull_build_images?id=buildpull-docker-images) about various options including images from [NVIDIA GPU Cloud](https://www.nvidia.com/en-us/gpu-cloud/). We may not support the newest framework versions due to API change.

## Quick start
Assuming CUDA enabled GPU is present, execute the following commands to run simple experiment with ResNet50 model  (if you do not have GPUs, see below):
```bash
# Go to DLBS home folder
cd dlbs
# Setup python paths
export PYTHONPATH=$(pwd)/python:$PYTHONPATH
# Build TensorFlow image. In the case of TensorFlow, the `hpe/tensorflow:cuda9-cudnn7` image
# located in tensorflow/cuda9-cudnn7 is the default TensorFlow image.
# Alternatively, you can skip this step and use your own image, pull image from NVIDIA GPU Cloud
# or use your bare metal TensorFlow installation.
# This will build an image named `hpe/tensorflow:cuda9-cudnn7`
cd ./docker
./build.sh tensorflow/cuda9-cudnn7
cd ..
# Create folder for experiment results
mkdir -p ./benchmarks/my_experiment
# Run experiment
python ./python/dlbs/experimenter.py run -Pexp.framework='"tensorflow"' -Pexp.model='"resnet50"' -Pexp.gpus='"0"' -Pexp.log_file='"./benchmarks/my_experiment/tf.log"'
# Print some results
python ./python/dlbs/logparser.py ./benchmarks/my_experiment/tf.log --output_params "exp.device_type,exp.phase,results.time,results.throughput,exp.framework_title,exp.model_title,exp.replica_batch,exp.framework_ver"
```

To use multiple GPUs with data parallel schema, provide list of GPUs i.e. `--exp.gpus='"0,1,2,3"'`
to use 4 GPUs. If you do not have NVIDIA GPUs, set list of GPUs to empty value i.e. `--exp.gpus='""'`. That will instruct
benchmarking suite to use CPUs.

If everything is OK, you should expect seeing JSON similar to this one:
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
            "results.time": 273.27,
            "results.throughput": 58.55
        }
    ]
}
```
The `results.time` - is an average time in milliseconds to process one batch of data.  If it is not there,
study ./benchmarks/my_experiment/tf.log for error messages. The `results.throughput` parameter is the number
of instances per second, in this case, number of images/seconds.

The [introduction](https://hewlettpackard.github.io/dlcookbook-dlbs/#/intro/intro?id=introduction-to-benchmarking-suite) contains more examples of what DLBS can do.

## Documentation

We host documentation [here](https://hewlettpackard.github.io/dlcookbook-dlbs/#/).

## More information

* [Why we created Benchmarking Suite](https://developer.hpe.com/blog/why-we-created-hpe-deep-learning-cookbook)
* GTC 2018 [presentation](http://on-demand.gputechconf.com/gtc/2018/video/S8555) / [slides](http://on-demand.gputechconf.com/gtc/2018/presentation/s8555-hpe-deep-learning-cookbook-recipes-to-run-deep-learning-workloads.pdf)
* [HPE Developer Portal](https://www.hpe.com/software/dl-cookbook)
* [HPE Deep Learning Performance Guide](http://dlpg.labs.hpe.com/)

## License

Deep Learning Benchmarking Suite is licensed under [Apache 2.0](../LICENSE) license.

## Contact us
* Natalia Vassilieva <nvassilieva@hpe.com>
* Sergey Serebryakov <sergey.serebryakov@hpe.com>
