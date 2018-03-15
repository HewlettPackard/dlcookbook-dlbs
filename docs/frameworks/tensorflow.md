
# __TensorFlow__

TensorFlow is integrated with benchmarking suite via tensorflow CNN benchmarks [projects](https://github.com/tensorflow/benchmarks). The git hash of the version we use is [4b337c13b1d71c67a9097779a2d92a9e1cc7ba2a](https://github.com/tensorflow/benchmarks/commit/4b337c13b1d71c67a9097779a2d92a9e1cc7ba2a).

## Standalone run
See [tf_cnn_benchmarks.py](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/tf_cnn_benchmarks/tf_cnn_benchmarks.py) for a complete list of supported command line arguments. Website with detailed description and explanation is located [here](https://www.tensorflow.org/performance/).


## Models
Complete list of supported models can be found on the [model](/models/models.md?id=supported-models) page.

## Adding new model
To add new model, several steps need to be performed:
1. Add a new file in [tf_cnn_benchmarks](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/python/tf_cnn_benchmarks) folder. Create a class and inherit it from `model.Model`.
2. Study any implementation of the existing model. The [DeepMNIST](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/tf_cnn_benchmarks/deepmnist_model.py) model is probably the simplest one.
3. Update [model_config.py](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/tf_cnn_benchmarks/model_config.py) file to return appropriate instance when requested.
4. Edit [tf_cnn_benchmarks.py](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/tf_cnn_benchmarks/tf_cnn_benchmarks.py) file. Basically, you need to search for `deep_mnist` string and see if you need to modify code accordingly. Several things may need to be changed:
    1. Number of classes (line 621).
    2. Number of input channels (line 1092).

## Commonly used configuration parameters
#### __tensorflow.docker_image__

* __default value__ `"hpe/tensorflow:cuda9-cudnn7"`
* __description__ The name of a docker image to use for TensorFlow if containerized benchmark is requested.

#### __tensorflow.host_libpath__

* __default value__ `""`
* __description__ Basically, it's a LD_LIBRARY_PATH for TensorFlow in case of a bare metal run.

#### __tensorflow.local_parameter_device__

* __default value__ `"cpu"`
* __description__ This is a 'local_parameter_device' parameter for tf_cnn_benchmarks. See tf_cnn_benchmarks.py for more details.

#### __tensorflow.use_nccl__

* __default value__ `True`
* __description__ This is a 'use_nccl' parameter for tf_cnn_benchmarks. See tf_cnn_benchmarks.py for more details.

#### __tensorflow.var_update__

* __default value__ `"replicated"`
* __description__ This is a 'variable_update' parameter for tf_cnn_benchmarks. See tf_cnn_benchmarks.py for more details.


## Other parameters
#### __tensorflow.args__

* __default value__ `[u'--model=${exp.model}', u'--eval=false', u"--forward_only=$('true' if '${exp.phase}'=='inference' else 'false')$", u'--batch_size=${exp.replica_batch}', u'--num_batches=${exp.num_batches}', u'--num_warmup_batches=${exp.num_warmup_batches}', u"--num_gpus=$(${exp.num_local_gpus} if '${exp.device_type}' == 'gpu' else 1)$", u'--display_every=1000', u'--device=${exp.device_type}', u"--data_format=$('NCHW' if '${exp.device_type}' == 'gpu' else 'NHWC')$", u'--variable_update=${tensorflow.var_update}', u"$('--use_nccl=${tensorflow.use_nccl}' if '${exp.device_type}' == 'gpu' else '--use_nccl=false')$", u'--local_parameter_device=${tensorflow.local_parameter_device}', u"$('' if not '${tensorflow.data_dir}' else '--data_dir=${tensorflow.data_dir}' if ${exp.docker} is False else '--data_dir=/workspace/data')$", u"$('--data_name=${tensorflow.data_name}' if '${tensorflow.data_name}' else '')$", u'--distortions=${tensorflow.distortions}', u'--num_intra_threads=${tensorflow.num_intra_threads}', u'--resize_method=${tensorflow.resize_method}']`
* __description__ These are a command line arguments passed to tf_cnn_benchmarks script.

#### __tensorflow.data_dir__

* __default value__ `""`
* __description__ A data directory if real data should be used. If empty, synthetic data is used \(no data ingestion pipeline\). See tf_cnn_benchmarks.py for more details.

#### __tensorflow.data_name__

* __default value__ `""`
* __description__ This is a 'data_name' parameter for tf_cnn_benchmarks. See tf_cnn_benchmarks.py for more details.

#### __tensorflow.distortions__

* __default value__ `False`
* __description__ This is a 'distortions' parameter for tf_cnn_benchmarks. See tf_cnn_benchmarks.py for more details.

#### __tensorflow.docker_args__

* __default value__ `[u'-i', u'--security-opt seccomp=unconfined', u'--pid=host', u'--volume=${DLBS_ROOT}/python/tf_cnn_benchmarks:/workspace/tf_cnn_benchmarks', u"$('--volume=${runtime.cuda_cache}:/workspace/cuda_cache' if '${runtime.cuda_cache}' else '')$", u"$('--volume=${tensorflow.data_dir}:/workspace/data' if '${tensorflow.data_dir}' else '')$", u"$('--volume=${monitor.pid_folder}:/workspace/tmp' if ${monitor.frequency} > 0 else '')$", u'${exp.docker_args}', u'${tensorflow.docker_image}']`
* __description__ In case if containerized benchmarks, this are the docker parameters.

#### __tensorflow.env__

* __default value__ `[u'PYTHONPATH=${tensorflow.python_path}:\\$PYTHONPATH', u'${runtime.EXPORT_CUDA_CACHE_PATH}', u'${runtime.EXPORT_CUDA_VISIBLE_DEVICES}']`
* __description__ Environmental variables to set for TensorFlow benchmarks.

#### __tensorflow.launcher__

* __default value__ `"${DLBS_ROOT}/scripts/launchers/tensorflow_hpm.sh"`
* __description__ Path to a script that launches TensorFlow benchmarks.

#### __tensorflow.num_intra_threads__

* __default value__ `0`
* __description__ This is a 'num_intra_threads' parameter for tf_cnn_benchmarks. See tf_cnn_benchmarks.py for more details.

#### __tensorflow.python_path__

* __default value__ `"$('${DLBS_ROOT}/python/tf_cnn_benchmarks' if ${exp.docker} is False else '/workspace/tf_cnn_benchmarks')$"`
* __description__ Path to a TensorFlow benchmarks python folder. Depends on if bare metal/docker based benchmark is requested.

#### __tensorflow.resize_method__

* __default value__ `"bilinear"`
* __description__ This is a 'resize_method' parameter for tf_cnn_benchmarks. See tf_cnn_benchmarks.py for more details.
