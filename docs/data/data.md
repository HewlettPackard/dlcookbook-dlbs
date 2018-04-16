# __Data__

DLBS can benchmark DL workloads with __synthetic__ or __real__ data. Synthetic data means there's no real dataset. Instead,
a synthetic (fake, dummy) dataset stored in memory is used. This basically means that with synthetic data we do not take
into account overhead associated with ingesting data. Why is synthetic data useful?

1. It gives an optimal performance assuming overhead associated with data ingestion pipeline is zero. Thus, it can be used to
   evaluate performance of ingestion pipelines from both software and hardware (storage) perspectives.
2. Data ingestion is infrastructure dependent. Data, depending on its size, can be stored in memory, local HDD/SDD, NFS or some
   high performance parallel/distributed file system. We do not know this in advance and wrong assumptions may lead to
   incorrect numbers that can be either too optimistic or too pessimistic.
3. As it was mentioned, synthetic data provides optimal performance assuming data ingestion is completely overlapped with
   forward/backward computations. This is a good way to benchmark ingestion libraries and various data storage options.

What needs to be taken into account when benchmarking ingestion pipelines with DLBS?
1. Some of the supported frameworks provide additional parameters for tuning ingestion pipelines such as, for instance,
   number of preprocessing or loader threads. Setting these parameters properly may have significant impact on performance,
   especially, if benchmarked models are not computationally expensive such as AlexNetOWT or fully connected neural nets.
2. Components that load data and preprocess it (scale, mirror etc.) may not be optimally written. DLBS tries to reuse as much
   as possible what's available in frameworks. In some cases, such as PyTorch, a custom data loader was written that loads
   data from Caffe's LMDB datasets and this can be improved.
3. Various preprocessing options significantly influence preprocessing time. Be default, DLBS uses minimal set of
   transformations including crop/scale and mirror. No heavy distortions are enabled.
4. In general, it's a good idea to benchmark only ingestion pipeline getting performance under these conditions (no
   computations involved). At this moment, only PyTorch backend provides this functionality.
5. The location of data may have impact on performance, especially for light models such as AlexNetOWT that require
   high ingress traffic to keep GPUs busy.
6. The data caching will have a very significant impact on performance. The very first time data is accessed it may
   get cached by an operating system (if data set is not large). Thus, the first epoch will be slow. The following
   epochs will be dramatically faster. This generally results in a fact that bencharkers need to understand what they
   benchmark. The possible strategy could be the following:
   1. Make an assumption on what dataset is used. If it's small/medium sized, assume data will be cached. Either run
      warm-up epoch to force operating system to cache your dataset or put it in /dev/shm.
   2. For large datasets make sure it's not cached. Either disable file system cache, or, make sure the data is removed
      from cache before running new epoch/benchmark ([dd](https://www.gnu.org/software/coreutils/manual/html_node/dd-invocation.html)
      utility can do that - search for _nocache_ there). In current version of DLBS, there is no option to invoke custom
      user callback before running a new epoch. Contact us if you need this.

By default, benchmarking suite uses synthetic data. There are three global and multiple framework-specific parameters
that affect ingestion pipelines:
  1. `exp.data_dir` A full path to a folder where data is stored. Caffe's forks use LMDB/LEVELDB datasets, TensorFlow
     uses files in tfrecord format, Caffe2 and PyTorch use LMDB datasets. MXNet uses recordio files. The backend for NVIDIA
     inference engine TensorRT does not support real data.
     Default value of this parameter is empty what means use synthetic data.
  2. `exp.data` (synthetic, real). By default, the value of this parameter is set by experimenter script. It is 'synthetic'
     if `exp.data_dir` is empty and 'real' otherwise. Can be used to search for experiments with real data.
  3. `exp.data_store` This is optional parameter that indicates what type of storage was used. It is a user defined string
     with no specific format that indicates storage properties. Benchmarks can introduce any other parameters they need to
     provide additional details in a more structured way.

> Only CNNs support real data. Other models, such as fully connected ones (DeepMNIST, AcousticModel) do not support
> real data and can only be used with synthetic data.

It was mentioned that the `exp.data_dir` parameter defines path to a dataset. It's OK to use this parameter if one framework
is benchmarked. If two or more frameworks are benchmarks in a same experiment, it may not be very convenient to add extension
sections that will define value for this parameter depending on the framework. In this case, users can use framework specific
dataset paths that look like this `${framework_family}.data_dir`: `tensorflow.data_dir`, `mxnet.data_dir`, `caffe2.data_dir` etc.
In this case no extensions are required. By default, the value of `exp.data_dir` parameter is set to be `"${${exp.framework}.data_dir}"`,
so, it will pick whatever dataset is specified for current active framework.

One thing to remember preparing benchmark dataset is that various models define their own shape for input images. For instance,
InceptionV3's input shape is `3x299x299` while ResNet50's input shape is `3x224x224`. The
[models](/models/models.md?id=supported-models) section provides detailed information on all supported models and their input shapes.

The following sections describe framework specific parameters. They are divided into three categories: (1) __mandatory__ that needs
to be specified to enable real data, (2) __optional__ that are optional and may be skipped and (3) __critical__ that can significantly
influence the performance. Normally, you want to try several values for critical parameters to see what works best for you for
your particular configuration. Default values should work OK for compute intensive models such as ResNet50 that does not require
large number of images per second.

### Caffe
Caffe can work with datasets stored in LMDB or LEVELDB databases.
1. Mandatory parameters
   * `caffe.data_dir=""` A data directory if real data should be used. If empty, synthetic data is used (no data ingestion pipeline).
   * `caffe.data_mean_file=""` In case of real data, specifies path to an image mean file.
   * `caffe.data_backend="LMDB"` In case of real data, specifies its storage backend ('LMDB' or 'LEVELDB').
2. Optional parameters
   * `caffe.mirror=true` In case of real data, specifies if 'mirrowing' should be applied.

### Caffe2
Caffe2 can work with datasets stored in LMDB database.
1. Mandatory parameters
   * `caffe2.data_dir=""` A data directory if real data should be used. If empty, synthetic data is used (no data ingestion pipeline).
   * `caffe2.data_backend="lmdb"` In case of real data, specifies its storage backend.
2. Critical parameters
   * `caffe2.num_decode_thread=1` Number of image decode threads when real dataset is used. For deep compute intensive models
      it can be as small as 1. For high throughput models such as AlexNetOWT it should be set to 6-8 threads for 4 V100 to
      provide ~ 9k images/second (depending on the model of your processor).

### MXNet
MXNet can work with datasets stored in \*.rec files.
1. Mandatory parameters
   * `mxnet.data_dir` A data directory if real data should be used. If empty, synthetic data is used (no data ingestion pipeline).
2. Critical parameters
   * `mxnet.preprocess_threads=4` Number preprocess threads for data ingestion pipeline when real data is used.
   * `mxnet.prefetch_buffer=10` Number of batches to prefetch (buffer size).

### PyTorch
PyTorch work with Caffe's LMDB datasets.
1. Mandatory parameters
   * `pytorch.data_dir=""` A data directory if real data should be used. If empty, synthetic data is used (no data ingestion pipeline).
   * `pytorch.data_backend="caffe_lmdb"` The type of dataset specified by *pytorch.data_dir*. Two datasets are supported. The first one
      is *caffe_lmdb*. This is exactly the same type of datasets that Caffe frameworks use. The second type is *image_folder* that can
      be read by a torchvision's [ImageFolder dataset](https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py#L72).
2. Optional parameters
   * `pytorch.data_shuffle=false` Enable/disable shuffling for both real and synthetic datasets.
3. Critical parameters
   * `pytorch.num_loader_threads=4` Number of worker threads to be used by data loader (for real datasets).

### TensorFlow
TensorFlow can work with datasets stored in \*.tfrecord files. Basically, experimenter exposes a subset of data-related parameters of
a tf_cnn_benchmarks project.
1. Mandatory parameters
   * `tensorflow.data_dir=""` A data directory if real data should be used. If empty, synthetic data is used (no data ingestion pipeline).
     See tf_cnn_benchmarks.py for more details.
   * `tensorflow.data_name=""` This is a 'data_name' parameter for tf_cnn_benchmarks. See tf_cnn_benchmarks.py for more details. If you use imagenet
     type of dataset, set it to "imagenet".
2. Critical parameters
   * `tensorflow.distortions=false` This is a 'distortions' parameter for tf_cnn_benchmarks. See tf_cnn_benchmarks.py for more details.
     This activates additional image transformations and will significantly decrease throughput that ingestion pipeline can provide.
