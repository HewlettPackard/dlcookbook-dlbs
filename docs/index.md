# __Deep Learning Benchmarking Suite__
Deep Learning Benchmarking Suite (DLBS) is a collection of command line tools for running consistent and reproducible deep learning benchmark experiments on various hardware/software platforms. In particular, DLBS:
1. Provides implementation of a number of neural networks in order to enforce apple-to-apple comparison across all supported frameworks. Models that are supported include various VGGs, ResNets, AlexNet and GoogleNet models. DLBS can support many more models via integration with third party benchmark projects such as Google's TF CNN Benchmarks or Tensor2Tensor.
2. Benchmarks single node multi-GPU or CPU  platforms. List of supported frameworks include various forks of Caffe (BVLC/NVIDIA/Intel), Caffe2, TensorFlow, MXNet, PyTorch. DLBS also supports NVIDIA's inference engine TensorRT for which DLBS provides highly optimized benchmark backend.
3. Supports inference and training phases.
4. Supports synthetic and real data.
5. Supports bare metal and docker environments.
6. Supports single/half/int8 precision and uses tensor cores with Volta GPUs.
7. Is based on modular architecture enabling easy integration with other projects
   such Google's TF CNN Benchmarks and Tensor2Tensor or NVIDIA's NVCNN, NVCNN-HVD or similar.
8. Supports `raw performance` metric (number of data samples per second like images/sec).

## Supported platforms
Deep Learning Benchmarking Suite was tested on various servers with Ubuntu / RedHat / CentOS operating systems with and without NVIDIA GPUs. We have a little success with running DLBS on top of AMD GPUs, but this is mostly untested. It may not work with Mac OS due to slightly different command line API of some of the tools we use (like, for instance, sed) - we will fix this in one of the next releases.

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
   > Our recommendation is to use docker images specified in default DLBS configuration. Most of them
   > are docker images from NVIDIA GPU Cloud.

## Quick start
Assuming CUDA enabled GPU is present, execute the following commands to run simple experiment with ResNet50 model:
```bash
git clone https://github.com/HewlettPackard/dlcookbook-dlbs.git ./dlbs   # Install benchmarking suite

cd ./dlbs  &&  source ./scripts/environment.sh                           # Initialize host environment
python ./python/dlbs/experimenter.py help --frameworks                   # List supported DL frameworks
docker pull nvcr.io/nvidia/tensorflow:18.07-py3                          # Pull TensorFlow docker image from NGC

python $experimenter run\                                                # Benchmark ...
       -Pexp.framework='"nvtfcnn"'\                                      #     TensorFlow framework
       -Vexp.model='["resnet50", "alexnet_owt"]'\                        #     with ResNet50 and AlexNetOWT models
       -Vexp.gpus='["0", "0,1", "0,1,2,3"]'\                             #     run on 1, 2 and 4 GPUs
       -Pexp.dtype='"float16"'                                           #     use mixed-precision training
       -Pexp.log_file='"${HOME}/dlbs/logs/${exp.id}.log"' \              #     and write results to these files

python $logparser '${HOME}/dlbs/logs/*.log'\                             # Parse log files and
       --output_file '${HOME}/dlbs/results.json'                         #     print and write summary to this file

python $reporter --summary_file '${HOME}/dlbs/results.json'\             # Parse summary file and build
                 --type 'weak-scaling'\                                  #     weak scaling report
                 --target_variable 'results.time'                        #     using batch time as performance metric
```

This configuration will run 6 benchmarks (2 models times 3 GPU configurations). DLBS can support multiple benchmark backends for Deep Learning frameworks. In this particular example DLBS uses a TensorFlow's `nvtfcnn` benchmark backend from NVIDIA which is optimized for single/multi-GPU systems. The introduction section contains more information on what backends actually represent and what users should be using.

The [introduction](https://hewlettpackard.github.io/dlcookbook-dlbs/#/intro/intro?id=introduction-to-benchmarking-suite) contains more examples of what DLBS can do.

## Documentation
We host documentation [here](https://hewlettpackard.github.io/dlcookbook-dlbs/#/).

## More information
* [Why we created Benchmarking Suite](https://developer.hpe.com/blog/why-we-created-hpe-deep-learning-cookbook)
* GTC 2018 [presentation](http://on-demand.gputechconf.com/gtc/2018/video/S8555) / [slides](http://on-demand.gputechconf.com/gtc/2018/presentation/s8555-hpe-deep-learning-cookbook-recipes-to-run-deep-learning-workloads.pdf)
* [HPE Developer Portal](https://www.hpe.com/software/dl-cookbook)
* [HPE Deep Learning Performance Guide](http://dlpg.labs.hpe.com/)

## License
Deep Learning Benchmarking Suite is licensed under [Apache 2.0](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/LICENSE) license.

## Contributing
All contributors must include acceptance of the DCO (Developer Certificate of Origin). Please, read this [document](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/CONTRIBUTING.md) for more details.

## Contact us
* Natalia Vassilieva <nvassilieva@hpe.com>
* Sergey Serebryakov <sergey.serebryakov@hpe.com>
