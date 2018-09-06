# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf
import horovod.tensorflow as hvd
import sys
from tensorflow.contrib.data.python.ops import interleave_ops
from tensorflow.contrib.data.python.ops import batching

def _deserialize_image_record(record):
    feature_map = {
        'image/encoded':          tf.FixedLenFeature([ ], tf.string, ''),
        'image/class/label':      tf.FixedLenFeature([1], tf.int64,  -1),
        'image/class/text':       tf.FixedLenFeature([ ], tf.string, ''),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32)
    }
    with tf.name_scope('deserialize_image_record'):
        obj = tf.parse_single_example(record, feature_map)
        imgdata = obj['image/encoded']
        label   = tf.cast(obj['image/class/label'], tf.int32)
        bbox    = tf.stack([obj['image/object/bbox/%s'%x].values
                            for x in ['ymin', 'xmin', 'ymax', 'xmax']])
        bbox = tf.transpose(tf.expand_dims(bbox, 0), [0,2,1])
        text    = obj['image/class/text']
        return imgdata, label, bbox, text

def _decode_jpeg(imgdata, channels=3):
    return tf.image.decode_jpeg(imgdata, channels=channels,
                                fancy_upscaling=False,
                                dct_method='INTEGER_FAST')

def _crop_and_resize_image(image, original_bbox, height, width, deterministic=False, random_crop=False):
    with tf.name_scope('random_crop_and_resize'):
        eval_crop_ratio = 0.8
        if random_crop:
            bbox_begin, bbox_size, bbox = \
                tf.image.sample_distorted_bounding_box(
                    tf.shape(image),
                    bounding_boxes=tf.zeros(shape=[1,0,4]), # No bounding boxes
                    min_object_covered=0.1,
                    aspect_ratio_range=[0.8, 1.25],
                    area_range=[0.1, 1.0],
                    max_attempts=100,
                    seed=7 * (1+hvd.rank()) if deterministic else 0,
                    use_image_if_no_bounding_boxes=True)
            image = tf.slice(image, bbox_begin, bbox_size)
        else:
            # Central crop
            image = tf.image.central_crop(image, eval_crop_ratio)
        image = tf.image.resize_images(
            image,
            [height, width],
            tf.image.ResizeMethod.BILINEAR,
            align_corners=False)
        image.set_shape([height, width, 3])
        return image

def _distort_image_color(image, order=0):
    with tf.name_scope('distort_color'):
        image = tf.multiply(image, 1. / 255.)
        brightness = lambda img: tf.image.random_brightness(img, max_delta=32. / 255.)
        saturation = lambda img: tf.image.random_saturation(img, lower=0.5, upper=1.5)
        hue        = lambda img: tf.image.random_hue(img, max_delta=0.2)
        contrast   = lambda img: tf.image.random_contrast(img, lower=0.5, upper=1.5)
        if order == 0: ops = [brightness, saturation, hue, contrast]
        else:          ops = [brightness, contrast, saturation, hue]
        for op in ops:
            image = op(image)
        # The random_* ops do not necessarily clamp the output range
        image = tf.clip_by_value(image, 0.0, 1.0)
        # Restore the original scaling
        image = tf.multiply(image, 255.)
        return image

def _parse_and_preprocess_image_record(record, counter, height, width,
                                       deterministic=False, random_crop=False,
                                       distort_color=False, nsummary=10):
    imgdata, label, bbox, text = _deserialize_image_record(record)
    label -= 1 # Change to 0-based (don't use background class)
    with tf.name_scope('preprocess_train'):
        try:    image = _decode_jpeg(imgdata, channels=3)
        except: image = tf.image.decode_png(imgdata, channels=3)

        # TODO: Work out a not-awful way to do this with counter being a Tensor
        #if counter < nsummary:
        #    image_with_bbox = tf.image.draw_bounding_boxes(
        #        tf.expand_dims(tf.to_float(image), 0), bbox)
        #    tf.summary.image('original_image_and_bbox', image_with_bbox)
        image = _crop_and_resize_image(image, bbox, height, width, deterministic, random_crop)
        #if counter < nsummary:
        #    tf.summary.image('cropped_resized_image', tf.expand_dims(image, 0))

        # image comes out of crop as float32, which is what distort_color expects
        if distort_color:
            image = _distort_image_color(image)
        image = tf.cast(image, tf.uint8)
        if random_crop:
            image = tf.image.random_flip_left_right(image,
                        seed=11 * (1 + hvd.rank()) if deterministic else None)
        #if counter < nsummary:
        #    tf.summary.image('flipped_image', tf.expand_dims(image, 0))
        return image, label

# Synthetic images are generated once, and the same batch is repeated again and
# again. The H2D copy is also repeated.
def fake_image_set(batch_size, height, width):
    data_shape = [batch_size, height, width, 3] # 3 channels
    images = tf.truncated_normal(
                 data_shape, dtype=tf.float32, mean=112, stddev=70,
                 name='fake_images')
    images = tf.clip_by_value(images, 0.0, 255.0)
    images = tf.cast(images, tf.uint8)
    labels = tf.random_uniform(
                 [batch_size], minval=0, maxval=1000-1, dtype=tf.int32,
                 name='fake_labels')
    images = tf.contrib.framework.local_variable(images, name='images')
    labels = tf.contrib.framework.local_variable(labels, name='labels')
    ds = tf.data.Dataset.from_tensor_slices(([images], [labels]))
    ds = ds.repeat()
    return ds

def image_set(filenames, batch_size, height, width, training=False,
              distort_color=False, num_threads=10, nsummary=10,
              deterministic=False):
    shuffle_buffer_size = 10000
    num_readers = 1
    ds = tf.data.Dataset.from_tensor_slices(filenames)
    if training:
        ds = ds.shard(hvd.size(), hvd.rank())
        ds = ds.repeat()
        ds = ds.shuffle(shuffle_buffer_size,
                 seed=5 * (1 + hvd.rank()) if deterministic else None)
    ds = ds.interleave(
        tf.data.TFRecordDataset, cycle_length=num_readers, block_length=1)
    counter = tf.data.Dataset.range(sys.maxsize)
    ds = tf.data.Dataset.zip((ds, counter))
    preproc_func = lambda record, counter_: _parse_and_preprocess_image_record(
        record, counter_, height, width, deterministic=deterministic,
        random_crop=training, distort_color=distort_color,
        nsummary=nsummary if training else 0)
    ds = ds.map(preproc_func, num_parallel_calls=num_threads)
    if training:
        ds = ds.shuffle(shuffle_buffer_size,
                 seed=13 * (1 + hvd.rank()) if deterministic else None)
    ds = ds.batch(batch_size)
    return ds

def image_set_new(filenames, batch_size, height, width,
                  training=False, distort_color=False,
                  deterministic=False,
                  num_threads=10, nsummary=10,
                  cache_data=False, num_splits=1):
    ds = tf.data.TFRecordDataset.list_files(filenames)
    if training:
        ds = ds.shard(hvd.size(), hvd.rank()) # HACK TESTING
    ds = ds.shuffle(buffer_size=10000,
             seed=5 * (1 + hvd.rank()) if deterministic else None)
    ds = ds.apply(interleave_ops.parallel_interleave(
        tf.data.TFRecordDataset, cycle_length=10))
    if cache_data:
        ds = ds.take(1).cache().repeat()
    counter = tf.data.Dataset.range(batch_size)
    counter = counter.repeat()
    ds = tf.data.Dataset.zip((ds, counter))
    ds = ds.prefetch(buffer_size=batch_size)
    if training:
        ds = ds.shuffle(buffer_size=10000,
                 seed=13 * (1 + hvd.rank()) if deterministic else None)
    ds = ds.repeat()
    preproc_func = lambda record, counter_: _parse_and_preprocess_image_record(
        record, counter_, height, width, deterministic,
        random_crop=training, distort_color=distort_color,
        nsummary=nsummary if training else 0)
    assert(batch_size % num_splits == 0)
    ds = ds.apply(
        batching.map_and_batch(
            map_func=preproc_func,
            batch_size=batch_size // num_splits,
            num_parallel_batches=num_splits))
    ds = ds.prefetch(buffer_size=num_splits)
    return ds
