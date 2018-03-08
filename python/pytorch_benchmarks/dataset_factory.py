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
"""PyTorch datasets to access various data formats:
    1. CaffeLMDBDataset - Caffe LMDB database.
    2. SyntheticDataset - synthetic data.
"""
from __future__ import absolute_import
from __future__ import print_function
import os
import pickle
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
try:
    import lmdb
    from PIL import Image
    from pytorch_benchmarks.caffe import datum_pb2
    HAVE_CAFFE_LMDB = True
except ImportError:
    HAVE_CAFFE_LMDB = False
    CAFFE_LMDB_EXCEPTION = "The Caffe LMDB dataset needs the following third-party "\
                           "dependencies: lmd, Pillow, protobuf. Install them with: "\
                           "'pip install Pillow lmdb protobuf'. Before installing "\
                           "lmdb, make sure you have wheel: 'pip install wheel'."


class CaffeLMDBDataset(data.Dataset):
    """
    https://github.com/pytorch/vision/blob/master/torchvision/datasets/lsun.py#L14
    https://developers.google.com/protocol-buffers/docs/pythontutorial
    https://stackoverflow.com/questions/33117607/caffe-reading-lmdb-from-python
    https://github.com/BVLC/caffe/blob/master/python/caffe/io.py
    """
    def __init__(self, db_path, transform=None):
        if not HAVE_CAFFE_LMDB:
            raise CAFFE_LMDB_EXCEPTION

        self.db_path = db_path
        self.env = lmdb.open(db_path, max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']
        cache_file = '_cache_' + db_path.replace('/', '_')
        if os.path.isfile(cache_file):
            self.keys = pickle.load(open(cache_file, "rb"))
        else:
            with self.env.begin(write=False) as txn:
                self.keys = [key for key, _ in txn.cursor()]
            pickle.dump(self.keys, open(cache_file, "wb"))
        self.transform = transform

    def __getitem__(self, index):
        """
        TODO: This probably needs to be optimized.
        """
        with self.env.begin(write=False) as txn:
            imgbuf = txn.get(self.keys[index])

        # Caffe stores data as CxHxW
        datum = datum_pb2.Datum()
        datum.ParseFromString(imgbuf)
        if len(datum.data) > 0:
            data = np.fromstring(datum.data, dtype=np.uint8).reshape(
                datum.channels, datum.height, datum.width
            )
            data = Image.fromarray(np.rollaxis(data, 0, 3))
        else:
            data = np.array(datum.float_data).astype(float).reshape(
                datum.channels, datum.height, datum.width
            )
            data = Image.fromarray(np.uint8(np.rollaxis(data, 0, 3)))

        if self.transform:
            data = self.transform(data)

        return data, datum.label

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


class SyntheticDataset(data.Dataset):
    """An implementation of a synthetic dataset. Provides random batch data and labels."""
    def __init__(self, size=100, shape=(3, 227, 227), num_classes=1000):
        """Initialize synthetic dataset

        :param int size: Size of a dataset.
        :param tuple shape: Shape of instances in this dataset,
        :param int num_classes: Number of classes in this dataset.
        """
        self.__size = size
        self.__shape = shape
        self.__num_classes = num_classes

        self.__data = np.random.uniform(-1, 1, (size,)+shape).astype(np.float32)
        self.__labels = np.random.randint(0, self.__num_classes, (size, 1)).astype(np.long)

    def shape(self):
        """Returns shape if instances."""
        return self.__shape

    def num_classes(self):
        """Returns number of classes."""
        return self.__num_classes

    def __len__(self):
        """Returns number of instances in this dataset

        :rtype: int
        "return: Number of instances in this dataset.
        """
        return self.__size

    def __getitem__(self, index):
        """Get index'th instance - a tuple of features and a label

        :param int index: Index of a dataset instance.
        :rtype: tuple
        :return: A tuple of (features, label)
        """
        instance = self.__data[index]
        label = self.__labels[index, 0]
        return instance, label


class DatasetFactory(object):
    """Creates various dataset loaders"""

    @staticmethod
    def get_data_loader(model, opts, batch_size):
        """Creates synthetic/real data loader.

        :param obj model: A model to benchmark
        :param dict opts: A dictionary of parameters.

        :return: An instance of data loader
        """
        if opts['data_dir'] == '':
            dataset = SyntheticDataset(size=opts['batch_size'] * 10,
                                       shape=model.module.input_shape,
                                       num_classes=model.module.num_classes)
        else:
            # Assuming (Channels, Height, Width). This is for image data now.
            # TODO: handle other types of data

            # https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py#L72
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            pipeline = [
                transforms.RandomResizedCrop(model.module.input_shape[-1]),
                transforms.ToTensor(),
                normalize,
            ]
            data_backend = opts['data_backend']
            if data_backend == 'image_folder':
                dataset = datasets.ImageFolder(opts['data_dir'], transforms.Compose(pipeline))
            elif data_backend == 'caffe_lmdb':
                dataset = CaffeLMDBDataset(opts['data_dir'], transforms.Compose(pipeline))
            else:
                raise ValueError("Invalid data backend (%s)" % data_backend)

        #TODO: is here effective batch size used?
        print("[WARNING] DataLoader - check that I accept effective batch size.")
        return data.DataLoader(
            dataset, batch_size=batch_size, shuffle=opts['data_shuffle'],
            num_workers=opts['num_loader_threads'], pin_memory=opts['num_gpus'] >= 1,
            sampler=None
        )
