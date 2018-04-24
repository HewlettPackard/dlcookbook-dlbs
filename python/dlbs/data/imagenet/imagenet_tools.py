# (c) Copyright [2017] Hewlett Packard Enterprise Development LP
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Various tools for ImageNet dataset that are used to assist generate
   framework-specific datasets. This file MUST NOT import framework-specific
   modules since it can be used in host OS.
"""
import os
import sys
import gzip
import json
import random
from dlbs.utils import IOUtils

class ImageNetTools(object):
    """Various framework-independent tools to process ImageNet and prepare files
       requried to generate benchmark datasets.
    """
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
    def get_image_files(folder, shuffle=True, num_files=-1):
        """ Get *.JPEG files in folder. Shuffle files and return at most num_files
            files.
        """
        # Scan the folder recursively and find files.
        files = IOUtils.find_files(folder, '*.JPEG', recursively=True)
        # Shuffle files and return first 'num_files' files.
        if shuffle:
            random.shuffle(files)
        if num_files > 0 and num_files < len(files):
            files = files[0:num_files]
        return files

    @staticmethod
    def get_file_info(img_file, labels):
        """Return meta information about image in 'img_file' file. """
        fdir, fname = os.path.split(img_file)
        synset = os.path.basename(fdir)
        if synset not in labels:
            raise ValueError("Invalid synset '%s: not found in labels dict." % synset)
        return synset, fname, labels[synset]

    @staticmethod
    def build_caffe_labels(imagenet_dir, labels_file):
        """Generates a textual file with the following content:
           img_0000.jpeg 1
           img_0001.jpeg 0
           ...
           mapping image file name to its class label
        """
        IOUtils.mkdirf(labels_file)
        img_files = ImageNetTools.get_image_files(imagenet_dir)
        labels = ImageNetTools.get_labels()
        with open(labels_file, 'w') as fobj:
            for img_file in img_files:
                synset, fname, finfo = ImageNetTools.get_file_info(img_file, labels)
                fobj.write("%s/%s %d\n" % (synset, fname, finfo['label']))

    @staticmethod
    def build_mxnet_labels(imagenet_dir, labels_file):
        """Generates a textual file with the following content:
           0   45  n02093256/n02093256_3032.JPEG
           1   45  n02093256/n02093256_3353.JPEG
           ...
           image_index   image_class_label   image_path
        """
        IOUtils.mkdirf(labels_file)
        img_files = ImageNetTools.get_image_files(imagenet_dir)
        labels = ImageNetTools.get_labels()
        with open(labels_file, 'w') as fobj:
            for img_index, img_file in enumerate(img_files):
                synset, fname, finfo = ImageNetTools.get_file_info(img_file, labels)
                fobj.write("%d\t%d\t%s/%s\n" % (img_index, finfo['label'], synset, fname))

    @staticmethod
    def build_tensorflow_synsets(imagenet_dir, synset_file):
        """Builds a textual file with one synset on a line"""
        IOUtils.mkdirf(synset_file)
        labels = ImageNetTools.get_labels()
        with open(synset_file, 'w') as fobj:
            for label in labels:
                fobj.write("%s\n" % label)

    @staticmethod
    def build_tensorflow_human_labels(imagenet_dir, human_labels_file):
        """Builds a textual file with one synset on a line"""
        IOUtils.mkdirf(human_labels_file)
        labels = ImageNetTools.get_labels()
        with open(human_labels_file, 'w') as fobj:
            for label in labels:
                fobj.write("%s\t%s\n" % (label, labels[label]['human_labels']))


if __name__ == '__main__':
    num_args = len(sys.argv)
    if num_args > 1:
        if sys.argv[1] == 'build_caffe_labels' and num_args == 4:
            ImageNetTools.build_caffe_labels(sys.argv[2], sys.argv[3])
        if sys.argv[1] == 'build_mxnet_labels' and num_args == 4:
            ImageNetTools.build_mxnet_labels(sys.argv[2], sys.argv[3])
        if sys.argv[1] == 'build_tensorflow_synsets' and num_args == 4:
            ImageNetTools.build_tensorflow_synsets(sys.argv[2], sys.argv[3])
        if sys.argv[1] == 'build_tensorflow_human_labels' and num_args == 4:
            ImageNetTools.build_tensorflow_human_labels(sys.argv[2], sys.argv[3])
