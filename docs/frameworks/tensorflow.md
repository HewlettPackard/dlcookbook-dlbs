
# __TensorFlow__

TensorFlow is integrated with benchmarking suite via tensorflow CNN benchmarks [projects](https://github.com/tensorflow/benchmarks). Currently, we do not provide support for the latest version. Version, bundled with benchmarking suite should be used instead.

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

### __tensorflow.var_update__ = `replicated`
This is a 'variable_update' parameter for tf_cnn_benchmarks. See tf_cnn_benchmarks.py for more details.
### __tensorflow.use_nccl__ = `true`
This is a 'use_nccl' parameter for tf_cnn_benchmarks. See tf_cnn_benchmarks.py for more details.
### __tensorflow.local_parameter_device__ = `cpu`
This is a 'local_parameter_device' parameter for tf_cnn_benchmarks. See tf_cnn_benchmarks.py for more details.
### __tensorflow.docker.image__ = `hpe/tensorflow:cuda9-cudnn7`
The name of a docker image to use for TensorFlow if containerized benchmark is requested.
### __tensorflow.host.libpath__ = `""`
Basically, it's a LD_LIBRARY_PATH for TensorFlow in case of a bare metal run.

## Other parameters

### __tensorflow.data_dir__ = `""`
A data directory if real data should be used. If empty, synthetic data is used
(no data ingestion pipeline). See tf_cnn_benchmarks.py for more details.
### __tensorflow.data_name__ = ``
This is a 'data_name' parameter for tf_cnn_benchmarks. See tf_cnn_benchmarks.py for more details.
### __tensorflow.distortions__ = `false`
This is a 'distortions' parameter for tf_cnn_benchmarks. See tf_cnn_benchmarks.py for more details.
### __tensorflow.num_intra_threads__ = `0`
This is a 'num_intra_threads' parameter for tf_cnn_benchmarks. See tf_cnn_benchmarks.py for more details.
### __tensorflow.resize_method__ = `bilinear`
This is a 'resize_method' parameter for tf_cnn_benchmarks. See tf_cnn_benchmarks.py for more details.

## Internal parameters

### __tensorflow.launcher__ = `${DLBS_ROOT}/scripts/launchers/tensorflow_hpm.sh`
Path to a script that launches TensorFlow benchmarks.
### __tensorflow.python_path__ = `$('${DLBS_ROOT}/python/tf_cnn_benchmarks' if '${exp.env}' == 'host' else '/workspace/tf_cnn_benchmarks')$`
Path to a TensorFlow benchmarks python folder. Depends on if bare metal/docker based benchmark is requested.
### __tensorflow.env__ = ...
Environmental variables to set for TensorFlow benchmarks.
### __tensorflow.args__ = ...
These are a command line arguments passed to tf_cnn_benchmarks script.",
### __tensorflow.docker.args__ = ...
In case if containerized benchmarks, this are the docker parameters.",
