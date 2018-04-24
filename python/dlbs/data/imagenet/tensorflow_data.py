"""
This file generates a simple benchmark dataset similar to the one used by Caffe.
It's based on Google's build_imagenet_data.py file.
"""
import os
import json
import random
import gzip
import multiprocessing
import argparse
from dlbs.utils import IOUtils

class Filesystem(object):
    @staticmethod
    def get_labels():
        """Load labels from 'imagenet_labels.json.gz' that's located in the same directory
           as this file.
           :rtype: dict
           :return: Dictionary that maps ImageNet class folder ID to an object that
                    contains two fields - 'label' and 'human_labels'. The 'label' is an
                    integer index of a label from [0,1000) according to this list:
                    http://data.csail.mit.edu/soundnet/categories/categories_imagenet.txt
        """
        labels_file = os.path.join(os.path.dirname(__file__), 'imagenet_labels.json.gz')
        with gzip.open(labels_file, 'rb') as file_obj:
            labels = json.load(file_obj)
        return labels
    
    @staticmethod
    def get_image_files(folder, num_files=-1):
        """ Get *.JPEG files in folder. Shuffle files and return at most num_files
            files.
        """
        # Scan the folder recursively and find files.
        files = IOUtils.find_files(folder, '*.JPEG', recursively=True)
        # Shuffle files and return first 'num_files' files.
        random.shuffle(files)
        if num_files > 0 and num_files < len(files):
            files = files[0:num_files]
        return files


def get_meta_file_metainfo(img_file, labels):
    """Return meta information about image in 'img_file' file. This information
       includes synset (str), numerical class label and human readeable set of
       class labels (comma separated str)
    """
    synset = img_file.split('/')[-2]
    if synset not in labels:
        raise ValueError("Invalid synset '%s: not found in labels dict." % synset)
    return (synset, labels[synset]['label'], str(labels[synset]['human_labels']))


def shard_files(files, num_shards, num_workers):
    """Split list of files ('files') into 'num_shards' shards of approximately equal
       size and assign those shards to 'num_workers' workers.
       :rtype: List of shard indices grouped by workers.
       :return: [[(),(),()],[(),(),()],[(),(),()]]
                   worker1   worker2     worker3
                () is a tuple of shard indices [first_index, last_index) for 'files'
                list.
    """
    num_files = len(files)
    # Compute shard index ranges. The ranges is [(), (), (), ..., ()]
    num_files_per_shard = num_files / num_shards
    shards = [None] * num_shards
    for shard_id in xrange(num_shards):
        shards[shard_id] = (num_files_per_shard*shard_id, num_files_per_shard*(shard_id+1))
    shards[-1] = (shards[-1][0], num_files)
    assert len(shards) == num_shards,\
           "Condition 'len(shards) == num_shards' failed (%d == %d) (shards: %s)" % (len(shards), num_shards, str(shards))

    # Here, we know that num_shards % num_workers is 0
    num_shards_per_worker = num_shards / num_workers
    shards = [shards[i:i+num_shards_per_worker] for i in xrange(0, len(shards), num_shards_per_worker)]
    assert len(shards) == num_workers,\
           "Condition 'len(shards) == num_workers' failed (%d == %d)" % (len(shards), num_workers)
    
    return shards


def create_tfrecord(files, shards, labels, worker_id, tfrecords_folder,
                    num_shards, num_workers, target_size=255):
    """A worker function. Worker needs to process shards 'shards' that define
       images in 'files' list.

       :param list files: List of all file names.
       :param list shards: List of shard indices to process by this worker
       :param dict labels: A dictionary that maps synset to labels
       :param in worker_id: Integer index of the worker. This also corresponds to
                            GPU index this worker will use.
       :param str tfrecords_folder: Folder to serialize tfrecord files.
       :param int num_shards: Total number of shards for all workers.
       :param int num_workers: Number of workers.
       :param int target_size: Target image size.
    """
    from dlbs.data.imagenet.tensorflow_worker import ImageCoder
    from dlbs.data.imagenet.tensorflow_worker import TFRecord
    import tensorflow as tf
    
    coder = ImageCoder(worker_id, target_size)
    for idx, shard in enumerate(shards):
        shard_index = (num_shards / num_workers)*worker_id + idx
        tfrecord_file = os.path.join(
            tfrecords_folder,
            'train-%.5d-of-%.5d' % (shard_index, num_shards)
        )
        writer = tf.python_io.TFRecordWriter(tfrecord_file)
        for img_file in files[shard[0]:shard[1]]:
            synset, label, human_label = get_meta_file_metainfo(img_file, labels)
            image = coder.decode_file(img_file)
            example = TFRecord.convert_to_example(
                img_file,
                image,
                label,
                synset,
                human_label,
                height=image.shape[0],
                width=image.shape[1]
            )
            writer.write(example.SerializeToString())


def main(images_folder, num_images, num_shards, tfrecords_folder,
         num_workers=1, target_size=255):
    #
    assert num_shards % num_workers == 0, "num_shards % num_workers must be 0"
    # Make sure 'tfrecords_folder' exists
    if not os.path.exists(tfrecords_folder):
        os.makedirs(tfrecords_folder)
    # Load labels
    labels = Filesystem.get_labels()
    # Get image files
    files = Filesystem.get_image_files(images_folder, num_images)
    shards = shard_files(files, num_shards, num_workers)
    print ("Shards: %s" % str(shards))
    # Start job processes
    workers = [None]*num_workers
    for worker_id in xrange(num_workers):
        workers[worker_id] = multiprocessing.Process(
            target=create_tfrecord,
            args=(files, shards[worker_id], labels, worker_id, tfrecords_folder,
                  num_shards, num_workers, target_size)
        )
        workers[worker_id].start()
    # Wait for completion
    for worker in workers:
        worker.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dir', type=str, required=True, default=None,
        help='Input directory with images (assuming ImageNet structure with valid synset identifiers).'
    )
    parser.add_argument(
        '--num_images', type=int, required=False, default=-1,
        help="Number of images to process. Images will be shuffled and then this number of "
             "images will be converted to tfrecords. Default value (-1) means convert all images."
    )
    parser.add_argument(
        '--num_shards', type=int, required=False, default=1,
        help="Number of shards (number of tfrecord files)."
    )
    parser.add_argument(
        '--output_dir', type=str, required=True, default=None,
        help='Output directory with tfrecord files.'
    )
    parser.add_argument(
        '--num_workers', type=int, required=False, default=1,
        help="Number of workers to process images. Must be less or equal to the number of GPUs. "
             "Also, num_shards % num_workers must be 0."
    )
    parser.add_argument(
        '--img_size', type=int, required=False, default=256,
        help="Target size of images in tfrecord files. Images will be (1) decoded from JPEG, "
             "(2) cropped relative to center along smallest dimension, (3) resized to this "
             "value and (4) casted back to uint8 data type. So, examples will contain 3D tensors "
             "of shape (img_size, img_size, 3) of type uint8 encoded as byte arrays."
    )
    args = parser.parse_args()
    
    main(images_folder=args.input_dir, num_images=args.num_images,
         num_shards=args.num_shards, tfrecords_folder=args.output_dir,
         num_workers=args.num_workers, target_size=args.img_size
    )
