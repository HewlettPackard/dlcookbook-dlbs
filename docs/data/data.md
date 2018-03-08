# __Data__

The benchmarking suite supports real and synthetic data. By default, synthetic data is used. Synthetic data is basically a randomly initialized tensor of an appropriate shape.

Two parameters are defined in `exp` namespace that can be used to provide an additional information to a processing scripts once benchmark has been done:

1. `exp.data` (synthetic, real). Indicates if real data was used. Must be provided by a user now. Default value is synthetic.
2. `exp.data_store` (mem, local-hdd, local-ssd, nfs ...) - a user defined value that described where data was located. Must be provided by a user.

Benchmarkers are welcome to introduce any other parameters they need to describe data ingestion pipeline in a more granular way.

For now, every framework has it's specific data ingestion parameters. However, a path to a dataset is always defined by a parameter `${framework_family}.data_dir`, for instance, `tensorflow.data_dir`, `caffe.data_dir`, `mxnet.data_dir` etc.

> TensorRT does not support real data - only synthetic.

> In current version, only image-type of datasets are supported. However, if input pipeline
> is only specified by a directory, it will work.

One thing to remember preparing benchmark dataset is that various models define their own shape for input images. For instance, InceptionV3's input shape is `3x299x299` while ResNet50's input shape is `3x224x224`. The [models](/models/models.md?id=supported-models) section provides detailed information on all supported models and their input shapes.

### Caffe
> Caffe can work with datasets stored in LMDB or LEVELDB databases.

1. `caffe.data_dir` A data directory if real data should be used. If empty, synthetic data is used (no data ingestion pipeline).
2. `caffe.mirror` In case of real data, specifies if 'mirrowing' should be applied.
3. `caffe.data_mean_file` In case of real data, specifies path to an image mean file."
4. `caffe.data_backend` In case of real data, specifies its storage backend ('LMDB' or 'LEVELDB').

### Caffe2
> Caffe2 can work with datasets stored in LMDB database.

1. `caffe2.data_dir` A data directory if real data should be used. If empty, synthetic data is used (no data ingestion pipeline).
2. `caffe2.data_backend` In case of real data, specifies its storage backend ('lmdb').

### MXNet
> Caffe2 can work with datasets stored in \*.rec files

1. `mxnet.data_dir` A data directory if real data should be used. If empty, synthetic data is used (no data ingestion pipeline).

### TensorFlow

> TensorFlow can work with datasets stored in \*.tfrecord files. Basically, experimenter
> exposes a subset of data-related parameters of a tf_cnn_benchmarks project.

1. `tensorflow.data_dir` A data directory if real data should be used. If empty, synthetic data is used (no data ingestion pipeline). See tf_cnn_benchmarks.py for more details.
2. `tensorflow.data_name` This is a 'data_name' parameter for tf_cnn_benchmarks. See tf_cnn_benchmarks.py for more details.
3. `tensorflow.distortions` This is a 'distortions' parameter for tf_cnn_benchmarks. See tf_cnn_benchmarks.py for more details.

> Setting `tensorflow.distortions` to true will significantly slow down easy computable
> models such as AlexNet.

### PyTorch
> PyTorch can now work with datasets of [raw images](http://pytorch.org/docs/master/torchvision/datasets.html#imagefolder).

1. `pytorch.data_dir` A data directory if real data should be used. If empty, synthetic data is used (no data ingestion pipeline).
2. `pytorch.data_backend` The type of dataset specified by *pytorch.data_dir*. Two datasets are supported. The first one is *caffe_lmdb*. This is exactly the same type of datasets that Caffe frameworks use. The second type is *image_folder* that can be read by a torchvision's [ImageFolder dataset](https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py#L72).
3. `pytorch.data_shuffle` Enable/disable shuffling for both real and synthetic datasets.
4. `pytorch.num_loader_threads` Number of worker threads to be used by data loader (for synthetic and real datasets).
