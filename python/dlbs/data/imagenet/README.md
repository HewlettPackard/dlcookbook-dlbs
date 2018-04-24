# Fast TensorFlow ingestion pipeline

### Dataset
First step to use fast ingestion pipeline with TensorFlow is the dataset generation.
The python file `tensorflow_data.py` located in this directory can do that.

__Input data__ is the standard ImageNet dataset. The dataset parent folder must
contain subfolders (synsets) and each subfolder must contain JPEG images of the
corresponding class. Assuming user has specified `exp.data_dir` parameter with
value '/path/to/imagenet', then '/path/to/imagenet' folder should contain subfolders
with names similar to 'n01440764' (ImageNet synsets). No other information is required.
The script will load `imagenet_labels.json.gz` file from current directory that maps
synsents to integer class labels.

__Output data__ is a collection of tfrecord files where each example has the
following features:
```python
example = tf.train.Example(features=tf.train.Features(feature={
    'image/height': TFRecord.int64_feature(height),
    'image/width': TFRecord.int64_feature(width),
    'image/colorspace': TFRecord.bytes_feature(colorspace),
    'image/channels': TFRecord.int64_feature(channels),
    'image/class/label': TFRecord.int64_feature(label),
    'image/format': TFRecord.bytes_feature(image_format),
    'image/encoded': TFRecord.bytes_feature(image.tobytes()),
    'image/filename': TFRecord.bytes_feature(os.path.basename(filename)),    
    'image/class/synset': TFRecord.bytes_feature(synset),
    'image/class/text': TFRecord.bytes_feature(human),
    }))
```
where `image/height`, `image/width` and `image/channels` are the height, width and
number of channels (always 3) in an image. The `image/class/label` is an integer class label,
`image/format` is always 'tensor', `image/colorspace` is always 'RGB', `image/class/synset`
is a synset UUID (something like 'n01440764'), `image/class/text` is a human readeable
class labels, `image/filename` is the original file name (without path) and `image/encoded`
is the actual image. The image is encoded as tensor of uint8 numerical values of shape
['image/height', 'image/width', 'image/channels'] i.e. the order is 'HWC' (channels last).

> In the future versions I will probably add ability to specify order - channels last or
> channels first. In this case, example instance will contain additional string feature
> 'image/order' with two possible values - 'HWC' and 'CHW'.

This is how one example can be parsed into image tensor and integer class label.
Pay attention, 'image' variable in the provided snippet is a Tensor instance with
HWC shape.
```python
  features = tf.parse_single_example(example_serialized, _feature_map)
  image = tf.decode_raw(features['image/encoded'], out_type=tf.uint8)
  height = tf.cast(features['image/height'], tf.int32)
  width = tf.cast(features['image/width'], tf.int32)
  channels = tf.cast(features['image/channels'], tf.int32)
  image_shape = tf.stack([height, width, channels])
  image = tf.reshape(image, image_shape)
  label = tf.cast(features['image/class/label'], dtype=tf.int32)
  return (image, label)
```

To run this tool to generate dataset, do the following (I do not have shell script
to run it in docker, so, I assume TensorFlow is installed in host OS):
```bash
# Go to DLBS root folder
cd /path/to/dlbs
# Set python paths
export PYTHONPATH=$(pwd)/python:$PYTHONPATH
# Run the tool
script=./python/dlbs/data/tensorflow_data.py
python $script --input_dir="/path/to/input/imagenet/dataset" \
               --num_images=-1 \
               --num_shards=10 \
               --output_dir="/path/to/output/folder/with/tfrecord/files" \
               --num_workers=1 \
               --img_size=300
```
Where:

1. `--num_images` Size of subset of entire dataset to convert in tfrecord format.
The value of '-1' means take all images. The tool will scan ImageNet folder, will
create list of all files, will shuffle it and then will take first __num_images__
files and will convert them in tfrecord files.
2. `--num_shards` Number of tfrecord files.
3. `--num_workers` Must be less or equal to the number of GPUs. I do not check this,
so make sure it's true.
4. `--img_size` The size of images in tfrecord dataset (img_size X img_size)

> DO NOT RUN it with full ImageNet  -  since it stores uncompressed tensors,
> the final size will be huge! 100,000 images of 300x300 resolution will require
> ~26G space. So you will need ~ 300G for entire ImageNet.

### Deep Learning Benchmarking Suite
The TF_CNN_BENCHMARK backend in this branch can load the dataset created above.
If I am not mistaken, the only modified file is `preprocessing.py` from tf_cnn_benchmarks
folder. To activate fast pipeline, you need to export the following variable:
```bash
export DLBS_TF_CNN_BENCHMARKS_FAST_PREPROCESSING=1
```
You can achieve this by providing the following parameter:
```json
{
  "parameters" {
    "runtime.launcher": "DLBS_TF_CNN_BENCHMARKS_FAST_PREPROCESSING=1"
  }
}
```

Most likey, the only net that will benefit from this pipeline is AlexNet.
