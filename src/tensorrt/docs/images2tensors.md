# Images2Tensors

### Overview
This tool converts images (JPEGs) to a binary representation that can directly be
used by the inference engine. This is a 'benchmark' dataset and it cannot be used
in any real use case. The goal is to have a collection of real images and completely
eliminate preprocessing by storing images in a format that does not require such a
preprocessing.

Images are stored as tensors of shape `[3, Width, Height]` where `Width` and `Height`
are always the same (`Size`). Each tensor is an array of length `3*Size*Size`. The
tool can create tensors of type `float` or `unsigned char` (`uchar`). For instance,
in case of unsigned char data type and images of shape [3,227,227], each image will
be represented as an array of 157587 elements (bytes) or 151KB per image.

Each output file can contain one or more tensors. Such files do not contain any
information on data types and exact shapes.

> In general, to emulate such binary file, it's just possible to create a file with
> random numbers of type 'unsigned char'. And all tools in this project will work
> just fine.

The directory structure of output files varies depending on how many images per
file a user wants to have. If this number is 1 (one file = one image), the tool
will replicate directory structure of input files keeping file names and extension.
If user wants to put multiple images in one output file, the directory structure
of output files will be flat with files named as `images-${shard}-${fileid}.tensors`
where `${shard}` is the integer identifier of a thread (multiple threads can be
used to speedup conversion) and `${fileid}` is file index in this shard.

Images2Tensor does not depend on TensorRT library and can run on hosts where
it's not available.

### Command line arguments
The tool is configured with the following command line arguments:
1. `--input_dir` Input directory. This directory must exist and must contain
images (jpg, jpeg) in that directory or one of its sub-directories. ImageNet
directory with raw images is one example of a valid directory structure.
2. `--output_dir` Output directory. The tool will write output files in this
directory.
3. `--size` Resize images to this size. Output images will have the following
shape [3, size, size].
4. `--dtype` A data type to use. Two types are supported: 'float' and 'uchar'. The
'float' is a single precision 4 byte numbers. Images take more space but are read
directly into an inference buffer. The 'uchar' (unsigned char) is a one byte
numbers that takes less disk space but need to be converted from unsigned char to
float array.
5. `--shuffle` Shuffle list of images. Is used with combination `--nimages` to
convert only a small random subset.
6. `--nimages` If nimages > 0, only convert this number of images. Use `--shuffle`
to randomly shuffle list of images with this option.
7. `--nthreads` Use this number of threads to convert images. This will significantly
increase overall throughput.
8. `--images_per_file` Number of images per output file.

For instance:
```bash
images2tensors --input_dir=/mnt/data/imagenet100k/jpegs \
               --output_dir=/mnt/data/imagenet100k/tensorrt \
               --size=227 --dtype=uchar --nthreads=5 --images_per_file=20000
```

### Build tensors dataset with DLBS
DLBS provides `make_imagenet_data.sh` script that can build datasets for various frameworks
including TensorRT:
```bash
source ./scripts/environment.sh
./scripts/make_imagenet_data.sh --dataset tensors1 \
                                --input /mnt/data/imagenet100k/jpegs \
                                --output /mnt/data/imagenet100k/tensorrt  \
                                --docker_image dlbs/tensorrt:18.10 \
                                --docker docker \
                                --images_per_file 20000 \
                                --num_workers 5
```
This script also accepts other parameters: `--images_per_file`, `--nimages` and `--shuffle`.
