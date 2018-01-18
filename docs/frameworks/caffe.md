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
#### __bvlc_caffe.docker_image__

* __default value__ `"hpe/bvlc_caffe:cuda9-cudnn7"`
* __description__ The name of a docker image to use for BVLC Caffe.

#### __bvlc_caffe.host_libpath__

* __default value__ `""`
* __description__ Basically, it's a LD_LIBRARY_PATH for BVLC Caffe in case of a bare metal run.

#### __bvlc_caffe.host_path__

* __default value__ `"${HOME}/projects/bvlc_caffe/build/tools"`
* __description__ Path to a BVLC Caffe executable in case of a bare metal run.

#### __intel_caffe.docker_image__

* __default value__ `"hpe/intel_caffe:cpu"`
* __description__ The name of a docker image to use for Intel Caffe.

#### __intel_caffe.host_libpath__

* __default value__ `""`
* __description__ Basically, it's a LD_LIBRARY_PATH for Intel Caffe in case of a bare metal run.

#### __intel_caffe.host_path__

* __default value__ `"${HOME}/projects/intel_caffe/build/tools"`
* __description__ Path to an Intel Caffe executable in case of a bare metal run.

#### __nvidia_caffe.docker_image__

* __default value__ `"hpe/nvidia_caffe:cuda9-cudnn7"`
* __description__ The name of a docker image to use for NVIDIA Caffe.

#### __nvidia_caffe.host_libpath__

* __default value__ `""`
* __description__ Basically, it's a LD_LIBRARY_PATH for NVIDIA Caffe in case of a bare metal run.

#### __nvidia_caffe.host_path__

* __default value__ `"${HOME}/projects/nvidia_caffe/build/tools"`
* __description__ Path to a NVIDIA Caffe executable in case of a bare metal run.


## Other parameters
#### __caffe.action__

* __default value__ `"$('train' if '${exp.phase}' == 'training' else 'time')$"`
* __description__ Action that needs to be performed by caffe. Possible values are 'train' or 'time'.

#### __caffe.args__

* __default value__ `[u"$('--solver=${caffe.model_dir}/${caffe.solver_file}' if '${exp.phase}' == 'training' else '')$", u"$('--model=${caffe.model_dir}/${caffe.model_file}' if '${exp.phase}' == 'inference' else '')$", u"$('-iterations ${exp.num_batches}' if '${exp.phase}' == 'inference' else '')$", u"$('--gpu=${exp.gpus}' if '${exp.device_type}' == 'gpu' else '')$"]`
* __description__ Command line arguments that launcher uses to launch Caffe.

#### __caffe.data_backend__

* __default value__ `"LMDB"`
* __description__ In case of real data, specifies its storage backend \('LMDB' or 'LEVELDB'\).

#### __caffe.data_dir__

* __default value__ `""`
* __description__ A data directory if real data should be used. If empty, synthetic data is used \(no data ingestion pipeline\).

#### __caffe.data_mean_file__

* __default value__ `""`
* __description__ In case of real data, specifies path to an image mean file.

#### __caffe.docker_args__

* __default value__ `[u'-i', u'--security-opt seccomp=unconfined', u'--pid=host', u'--volume=${DLBS_ROOT}/models/${exp.model}:/workspace/model', u"$('--volume=${runtime.cuda_cache}:/workspace/cuda_cache' if '${runtime.cuda_cache}' else '')$", u"$('--volume=${caffe.data_dir}:/workspace/data' if '${caffe.data_dir}' else '')$", u"$('--volume=${monitor.pid_folder}:/workspace/tmp' if ${monitor.frequency} > 0 else '')$", u'${exp.docker_args}', u'${caffe.docker_image}']`
* __description__ In case if containerized benchmarks, this are the docker parameters.

#### __caffe.docker_image__

* __default value__ `"${${caffe.fork}_caffe.docker_image}"`
* __description__ The name of a docker image to use.

#### __caffe.env__

* __default value__ `"${runtime.EXPORT_CUDA_CACHE_PATH}"`
* __description__ An string that defines export variables that's used to initialize environment for Caffe.

#### __caffe.fork__

* __default value__ `"bvlc"`
* __description__ A fork of Caffe. Possible values may vary including 'bvlc', 'nvidia' and 'intel'. The value of this parameter is computed automatically. The only reason why it has default value \(bvlc\) is because there are other parameters referencing this parameter that need to be resolved.

#### __caffe.host_path__

* __default value__ `"${${caffe.fork}_caffe.host_path}"`
* __description__ Path to a caffe executable in case of bare metal run.

#### __caffe.launcher__

* __default value__ `"${DLBS_ROOT}/scripts/launchers/caffe.sh"`
* __description__ Path to script that launches Caffe benchmarks.

#### __caffe.mirror__

* __default value__ `True`
* __description__ In case of real data, specifies if 'mirrowing' should be applied.

#### __caffe.model_dir__

* __default value__ `"$('${DLBS_ROOT}/models/${exp.model}' if not ${exp.docker} else '/workspace/model')$"`
* __description__ Directory where Caffe's model file is located. Different for host/docker benchmarks.

#### __caffe.model_file__

* __default value__ `"${exp.id}.model.prototxt"`
* __description__ Caffe's prototxt model file.

#### __caffe.solver__

* __default value__ `[u"net: '${caffe.model_dir}/${caffe.model_file}'\\n", u'max_iter: ${exp.num_batches}\\n', u'test_interval: 0\\n', u'snapshot: 0\\n', u'snapshot_after_train: false\\n', u'base_lr: 0.01\\n', u"lr_policy: 'fixed'\\n", u"solver_mode: $('${exp.device_type}'.upper())$\\n", u"$('solver_data_type: ${nvidia_caffe.solver_precision}' if '${exp.framework}' == 'nvidia_caffe' else '')$"]`
* __description__ A content for a Caffe's solver file in case Caffe benchmarks train phase.

#### __caffe.solver_file__

* __default value__ `"${exp.id}.solver.prototxt"`
* __description__ Caffe's prototxt solver file.

#### __nvidia_caffe.backward_math_precision__

* __default value__ `"$('FLOAT' if '${nvidia_caffe.precision}' in ('float32', 'mixed') else 'FLOAT16')$"`
* __description__ Precision for a backward math \(FLOAT, FLOAT16\). Only for NVIDIA Caffe. More details are here: http://on-demand.gputechconf.com/gtc/2017/presentation/s7218-training-with-mixed-precision-boris-ginsburg.pdf

#### __nvidia_caffe.backward_precision__

* __default value__ `"$('FLOAT16' if '${nvidia_caffe.precision}' in ('float16', 'mixed') else 'FLOAT')$"`
* __description__ Precision for a backward pass \(FLOAT, FLOAT16\). Only for NVIDIA Caffe. More details are here: http://on-demand.gputechconf.com/gtc/2017/presentation/s7218-training-with-mixed-precision-boris-ginsburg.pdf

#### __nvidia_caffe.forward_math_precision__

* __default value__ `"$('FLOAT' if '${nvidia_caffe.precision}' in ('float32', 'mixed') else 'FLOAT16')$"`
* __description__ Precision for a forward math \(FLOAT, FLOAT16\). Only for NVIDIA Caffe. More details are here: http://on-demand.gputechconf.com/gtc/2017/presentation/s7218-training-with-mixed-precision-boris-ginsburg.pdf

#### __nvidia_caffe.forward_precision__

* __default value__ `"$('FLOAT16' if '${nvidia_caffe.precision}' in ('float16', 'mixed') else 'FLOAT')$"`
* __description__ Precision for a forward pass \(FLOAT, FLOAT16\). Only for NVIDIA Caffe. More details are here: http://on-demand.gputechconf.com/gtc/2017/presentation/s7218-training-with-mixed-precision-boris-ginsburg.pdf

#### __nvidia_caffe.precision__

* __default value__ `"$('float32' if '${exp.dtype}' in ('float', 'float32', 'int8') else 'float16')$"`
* __description__ Parameter that specifies what components in NVIDIA Caffe use what precision:     float32   Use FP32 for training values storage and matrix-mult accumulator.               Use FP32 for master weights.     float16   Use FP16 for training values storage and matrix-mult accumulator               Use FP16 for master weights.     mixed     Use FP16 for training values storage and FP32 for matrix-mult accumulator               Use FP32 for master weights. More fine-grained control over these values can be done by directly manipulating the following parameters:      nvidia_caffe.solver_precision          Master weights     nvidia_caffe.forward_precision         Training values storage     nvidia_caffe.backward_precision        Training values storage     nvidia_caffe.forward_math_precision    Matrix-mult accumulator     nvidia_caffe.backward_math_precision   Matrix-mult accumulator Default value depends on exp.dtype parameters:     exp.dtype == float32 -> nvidia_caffe.precision = float32     exp.dtype == float16 -> nvidia_caffe.precision = float16     exp.dtype == int32   -> nvidia_caffe.precision is set to float32 and experiment will notbe ran For information:     http://on-demand.gputechconf.com/gtc/2017/presentation/s7218-training-with-mixed-precision-boris-ginsburg.pdf     https://github.com/NVIDIA/caffe/issues/420

#### __nvidia_caffe.solver_precision__

* __default value__ `"$('FLOAT' if '${nvidia_caffe.precision}' in ('float32', 'mixed') else 'FLOAT16')$"`
* __description__ Precision for a solver \(FLOAT, FLOAT16\). Only for NVIDIA Caffe. More details are here: http://on-demand.gputechconf.com/gtc/2017/presentation/s7218-training-with-mixed-precision-boris-ginsburg.pdf
