# Datasets

Inference benchmarks can be run with synthetic or real input datasets. Real
datasets can be of two types - (1) collection of images similar to ImageNet or
(2) a collection of binary files containing preprocessed images.

### Synthetic data
Synthetic data is basically random tensors in host memory. Instead of streaming
real images and writing them into input tensors, those tensors are randomly
initialized in the beginning and do not change. This implies that there's no
overhead associated with input data ingestion pipeline and storage/network.
The input tensors are allocated in host memory (depending on parameters, in host
pinned memory) what results in CPU-GPU traffic via PCIe lanes what results in a
fact that PCIe can actually be a bottleneck for high throughput models such as
AlexNetOWT. Depending on inference module being used, the benchmark tool writes
into log files inference times assuming data is in GPU memory and inference time
including CPU <--> GPU overhead. That makes it possible to identify if PCIe is
actually a bottleneck. These types of benchmarks can be useful from different
points of view:

1. Benchmark compute devices.
2.  Benchmark host to device transfers.

Running neural networks of different comptue complexity (ResNet152/50 vs AlexNetOWT)
will give you sort of compute profile of your system.


### Real data
__Images__. The benchmark tool can read ImageNet-like datasets stored as raw
images. This, however, may be slow due to:

1. Benchmark tool needs to read a large number of small files.
2. Images need to be resized and converted to floating point arrays.
3. We do not really optimize code in this particular scenario and use a
   straightforward implementation utilizing OpenCV library from Ubuntu repository.
   The benchmark tool pre-processes images on CPUs.

This type of dataset is disabled by default due to performance issues. Even if
a user provides this data, the benchmark tool will refuse to run unless environment
variable `DLBS_TENSORRT_ALLOW_IMAGE_DATASET` is set to true. For more details,
read documentation for the `Environment` class defined in `src/core/utils.hpp`.


__Tensors__. This is a special binary format that stores preprocessed images as
floating point or unsigned char arrays. We provide an [images2tensors](images2tensors.md)
tool to convert JPEG images into this format. This is sort of artificial format
that's used to store preprocessed images that can directly be streamed into
inference buffer not doing expensive preprocessing. The goal is basically to put
as much stress as possible to a storage/network side and to not consider a CPU as
a bottleneck. For more details, read documentation for the `Environment` class
defined in `src/core/utils.hpp`.
