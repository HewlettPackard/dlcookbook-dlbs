
# MXNet

We've integrated mxnet into benchmarking suite using similar to TensorFlow approach. We've written [mxnet_benchmarks](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/python/mxnet_benchmarks) python projects that exposes similar command line API as tf_cnn_benchmarks. This is a quick overview of this implementation.

## Standalone run
The project itself is located in [mxnet_benchmarks](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/python/mxnet_benchmarks) folder. The [benchmarks.py](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/mxnet_benchmarks/benchmarks.py) file is the entry point. Following command line parameters are supported:
1. `--model` (type=str) A model to benchmark ("alexnet", "googlenet" ...).
2. `--forward_only` (type=str) Benchmark inference (if true) else benchmark training.
3. `--batch_size` (type=int) Per device batch size.
4. `--num_batches` (type=int) Number of benchmark iterations.
5. `--num_warmup_batches` (type=int) Number of warmup iterations.
6. `--num_gpus` (type=int) Number of gpus to use (per node?). Use CUDA_VISIBLE_DEVICES to select those devices.
7. `--device` (type=str) Comptue device, "cpu" or "gpu".
8. `--kv_store` (type=str) Type of gradient aggregation schema (local, device, dist_sync, dist_device_sync, dist_async). See this [page](https://mxnet.incubator.apache.org/how_to/multi_devices.html) for more details.
9. `--data_dir` (type=str) Path to the image RecordIO (.rec) file or a directory path. Created with tools/im2rec.py. If not set or empty, synthetic data is used.


## Models
The class `ModelFactory` in [model_factory.py](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/mxnet_benchmarks/model_factory.py) is responsible for creating neural networks. Complete list of supported models can be found on the [model](/models/models.md?id=supported-models) page.

## Adding new model
To add new model, several steps need to be performed:
1. Add a new file in [models](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/python/mxnet_benchmarks/models) folder. Create a class and inherit it from [mxnet_benchmarks.models.model.Model](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/mxnet_benchmarks/models/model.py).
2. Study any implementation of the existing model. The [DeepMNIST](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/mxnet_benchmarks/models/deep_mnist.py) model is probably the simplest one.
    1. The model class must define `output` property that returns output of a head node (usually, `mx.symbol.SoftmaxOutput` for training and `mx.symbol.softmax` for inference). There is a helper method in base model class named `add_head_nodes` that adds final dense and softmax nodes for you.
    2. The constructor of a model class accepts one parameter - `params` - a dictionary that defines model property. In particular, it will contain at least these parameters: `model` (model id) and `phase` (`training` or `inference`). You then must ensure this dictionary also contains `name` (model name), `input_shape` (shape of input tensor excluding batch dimension) and `num_classes` (number of output classes). Base class then is initialized with parameters. A constructor must also construct the model itself.
