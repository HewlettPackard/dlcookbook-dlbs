
# TensorFlow

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
