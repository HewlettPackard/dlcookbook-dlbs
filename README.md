# __Deep Learning Benchmarking Suite__
Deep Learning Benchmarking Suite (DLBS) is a set of command line tools for providing consistent and reproducible benchmark experiments on various hardware/software combinations. In particular, DLBS provides the following functionality:
1. Implements internally various deep models. Our goal is to provide same model implementations for all supported frameworks. Deep models that are supported include various VGGs, ResNets, AlexNet and GoogleNet models.
2. Benchmarks single node multi GPU configurations. Frameworks that are now supported: BVLC Caffe, NVIDIA Caffe, Intel Caffe, Caffe2, TensorFlow, MXNet and TensorRT.
3. Supports inference and training phases.
4. Can use real data if dataset is available. Else, falls back to synthetic data.
5. Supports bare metal and docker environments.

## Supported platforms
Deep Learning Benchmarking Suite was tested on various servers with Ubuntu /
RedHat / CentOS operating systems with/without NVIDIA GPUs. It may not work with
Mac OS due to slightly different command line API of some of the tools we use
(like, for instance, sed) - we will fix this in one of the next releases.

## Installation
1. Install Docker and NVIDIA Docker for containerized benchmarks. Read [here](/docker/docker.md?id=docker) why we prefer to use docker and [here](/docker/install_docker.md?id=installing-docker) for installing/troubleshooting tips. This is not required. DLBS can work with bare metal framework installations.
2. Clone Deep Learning Benchmarking Suite from [GitHub](https://github.hpe.com/labs/dlcookbook.git)
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
# Setup python paths
export PYTHONPATH=$(pwd)/python:$PYTHONPATH
# Create folder for experiment results
mkdir -p ./benchmarks/my_experiment
# Run experiment
python ./python/dlbs/experimenter.py run -Pexp.framework='"tensorflow"' -Pexp.model='"resnet50"' -Pexp.gpus='"0"' -Pexp.bench_root='"./benchmarks/my_experiment"' -Pexp.log_file='"${exp.bench_root}/tf.log"'
# Print some results
python ./python/dlbs/logparser.py --keys exp.device results.training_time exp.framework_title exp.model_title exp.device_batch -- ./benchmarks/my_experiment/tf.log
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
python ./python/dlbs/experimenter.py run -Pexp.framework='"tensorflow"' -Pexp.model='"resnet50"' -Pexp.gpus='""' -Pexp.bench_root='"./benchmarks/my_experiment"' -Pexp.log_file='"${exp.bench_root}/tf.log"'
# Print some results
python ./python/dlbs/logparser.py --keys exp.device results.training_time exp.framework_title exp.model_title exp.device_batch -- ./benchmarks/my_experiment/tf.log
```

If everything is OK, you should expect seeing this JSON (training time - an average batch time - of course will be different):
```json
{
    "data": [
        {
            "exp.device": "gpu",
            "exp.device_batch": "16",
            "exp.framework_title": "TensorFlow",
            "exp.model_title": "ResNet50",
            "results.training_time": 255.59105431309905
        }
    ]
}
```

If `results.training_time` is not there, study ./benchmarks/my_experiment/tf.log for error messages.



## Deep Learning CookBook
Deep Learning Benchmarking Suite is part of HPE's Deep Learning CookBook project.
Read this [blog post](https://community.hpe.com/t5/Behind-the-scenes-Labs/The-Deep-Learning-Cookbook/ba-p/6967323#.WUmLVOvythE) about it.

## Documentation

We host documentation on GitHub pages [here](http://hewlettpackard.github.io/dlcookbook-dlbs).

## License

Deep Learning Benchmarking Suite is released under the [Apache 2.0 license](./LICENSE).

## Contact us

* Natalia Vassilieva <nvassilieva@hpe.com>
* Sergey Serebryakov <sergey.serebryakov@hpe.com>
