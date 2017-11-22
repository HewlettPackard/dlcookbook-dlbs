# __Caffe__

Various forks of Caffe are integrated into deep learning benchmarking suite. This integration is pretty straightforward. We use standard caffe's time and train command to benchmark training/inference. A launcher shell script, [Caffe wrapper](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/scripts/launchers/caffe.sh), exists that serves as mediator taking experiment parameters and invoking caffe.

## Standalone run
As is, there is no special tool to run caffe from command line. Use caffe directly passing all parameters that it accepts:

1. `--solver` A path to a solver prototxt descriptor when benchmarking phase is training.
2. `--model` A path to a model prototxt descriptor when benchmarking phase is inference.
3. `-iterations` Number of benchmarking iterations when benchmarking phase is inference.
4. `--gpu` Comma separated list of GPU ids to use of device is GPU.

## Models
For Caffe's forks, all models are stored in [models](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/models) directory in the root folder of the project.
Complete list of supported models can be found on the [model](/models/models.md?id=supported-models) page.

## Adding new model
To add new model, several steps need to be performed:
1. Create new folder with a model id name in [models](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/models).
2. Create there two files. One is a caffe training prototxt file ending with `.training.prototxt`. The other one is a Caffe deployment (inference) prototxt descriptor ending with `.inference.prototxt`.
3. Standard Caffe training descriptor must be modified in order to (1) support synthetic and real data, (2) support configurable batch size and (3) support float16/float32 inference in NVIDIA Caffe:
    1. Supporting synthetic/real data with configurable batch size. This is an example of input data specification that enables this functionality:

       ```bash
       #synthetic layer {
       #synthetic   name: "data"
       #synthetic   type: "Input"
       #synthetic   top: "data"
       #synthetic   input_param { shape: { dim: __EXP_DEVICE_BATCH__ dim: 3 dim: 227 dim: 227 } }
       #synthetic }
       #synthetic layer {
       #synthetic   name: "label"
       #synthetic   type: "Input"
       #synthetic   top: "label"
       #synthetic   input_param { shape: { dim: __EXP_DEVICE_BATCH__ dim: 1 } }
       #synthetic }

       #data layer {
       #data   name: "data"
       #data   type: "Data"
       #data   top: "data"
       #data   top: "label"
       #data   include {
       #data     phase: TRAIN
       #data   }
       #data   transform_param {
       #data     mirror: __CAFFE_MIRROR__
       #data     crop_size: 227
       #data     mean_file: "__CAFFE_DATA_MEAN_FILE__"
       #data   }
       #data   data_param {
       #data     source: "__CAFFE_DATA_DIR__"
       #data     batch_size: __EXP_DEVICE_BATCH__
       #data     backend: __CAFFE_DATA_BACKEND__
       #data   }
       #data }
       ```
    2. Supporting float16/float32 types for NVIDIA Caffe. Copy-paste these lines in the beginning of the training descriptor:

       ```bash
       #precision default_forward_type: __FORWARD_TYPE___
       #precision default_backward_type: __BACKWARD_TYPE___
       #precision default_forward_math: __FORWARD_MATH___
       #precision default_backward_math: __BACKWARD_MATH___
       ```
For simple examples, study deep MNIST [deployment](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/models/deep_mnist/deep_mnist.inference.prototxt) and [training](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/models/deep_mnist/deep_mnist.training.prototxt) descriptors.

## Commonly used configuration parameters

### __bvlc_caffe.host.path__ = `${HOME}/projects/bvlc_caffe/build/tools`
Path to a BVLC Caffe executable in case of a bare metal run.
### __bvlc_caffe.host.libpath__ = ``
Basically, it's a LD_LIBRARY_PATH for BVLC Caffe in case of a bare metal run.
### __bvlc_caffe.docker.image__ = `hpe/bvlc_caffe:cuda9-cudnn7`
The name of a docker image to use for BVLC Caffe.
### __nvidia_caffe.host.path__ = `${HOME}/projects/nvidia_caffe/build/tools`
Path to a NVIDIA Caffe executable in case of a bare metal run.
### __nvidia_caffe.host.libpath__ = ``
Basically, it's a LD_LIBRARY_PATH for NVIDIA Caffe in case of a bare metal run.
### __nvidia_caffe.docker.image__ = `hpe/nvidia_caffe:cuda9-cudnn7`
The name of a docker image to use for NVIDIA Caffe.
### __intel_caffe.host.path__ = ``
Path to an Intel Caffe executable in case of a bare metal run.
### __intel_caffe.host.libpath__ = `${HOME}/projects/intel_caffe/build/tools`
Basically, it's a LD_LIBRARY_PATH for Intel Caffe in case of a bare metal run.
### __intel_caffe.docker.image__ = `hpe/intel_caffe:cpu`
The name of a docker image to use for Intel Caffe.

## Other parameters

### __caffe.data_dir__ = ``
A data directory if real data should be used. If empty, synthetic data is used
(no data ingestion pipeline).
### __caffe.mirror__ = `true`
In case of real data, specifies if 'mirrowing' should be applied.
### __caffe.data_mean_file__ = ``
In case of real data, specifies path to an image mean file.",
### __caffe.data_backend__ = `LMDB`
In case of real data, specifies its storage backend ('lmdb' or 'leveldb').
### __nvidia_caffe.solver_precision__ = `FLOAT`
Precision for a solver (`FLOAT`, `FLOAT16`). Only for NVIDIA Caffe.
More details are [here](http://on-demand.gputechconf.com/gtc/2017/presentation/s7218-training-with-mixed-precision-boris-ginsburg.pdf)
### __nvidia_caffe.forward_precision__ = `FLOAT`
"Precision for a forward pass (`FLOAT`, `FLOAT16`). Only for NVIDIA Caffe.
More details are [here](http://on-demand.gputechconf.com/gtc/2017/presentation/s7218-training-with-mixed-precision-boris-ginsburg.pdf)
### __nvidia_caffe.forward_math_precision__ = `FLOAT`
Precision for a forward math (`FLOAT`, `FLOAT16`). Only for NVIDIA Caffe.
More details are [here](http://on-demand.gputechconf.com/gtc/2017/presentation/s7218-training-with-mixed-precision-boris-ginsburg.pdf)
### __nvidia_caffe.backward_precision__ = `FLOAT`
Precision for a backward pass (`FLOAT`, `FLOAT16`). Only for NVIDIA Caffe.
More details are [here](http://on-demand.gputechconf.com/gtc/2017/presentation/s7218-training-with-mixed-precision-boris-ginsburg.pdf)
### __nvidia_caffe.backward_math_precision__ = `FLOAT`
Precision for a backward math (`FLOAT`, `FLOAT16`). Only for NVIDIA Caffe.
More details are [here](http://on-demand.gputechconf.com/gtc/2017/presentation/s7218-training-with-mixed-precision-boris-ginsburg.pdf)

## Internal parameters

### __caffe.launcher__ = `${DLBS_ROOT}/scripts/launchers/caffe.sh`
Path to script that launches Caffe benchmarks.
### __caffe.fork__ = `bvlc`
A fork of Caffe. Possible values may vary including `bvlc`, `nvidia` and `intel`.
Will be set automatically by one of the extension modules.
### __caffe.phase__ = `$('train' if '${exp.phase}' == 'training' else 'deploy')$`
Caffe's benchmark phase. Possible values are `train` and `deploy`.
### __caffe.action__ = `$('train' if '${exp.phase}' == 'training' else 'time')$`
Action that needs to be performed by caffe. Possible values are `train` or `time`.
### __caffe.model_file__ = `${exp.id}.model.prototxt`
Caffe's prototxt model file.
### __caffe.solver_file__ = `${exp.id}.solver.prototxt`
Caffe's prototxt solver file.
### __caffe.model_dir__ = `$('${DLBS_ROOT}/models/${exp.model}' if '${exp.env}' == 'host' else '/workspace/model')$`
Directory where Caffe's model file is located. Different for host/docker benchmarks.
### __caffe.solver__ = ...
A content for a Caffe's solver file in case Caffe benchmarks train phase.",
### __caffe.args__ = ...
Command line arguments that launcher uses to launch Caffe.
### __caffe.host.path__ = `${${caffe.fork}_caffe.host.path}`
Path to a caffe executable in case of bare metal run.
### __caffe.docker.image__ = `${${caffe.fork}_caffe.docker.image}`
The name of a docker image to use.
### __caffe.docker.args__ = ...
In case if containerized benchmarks, this are the docker parameters.
