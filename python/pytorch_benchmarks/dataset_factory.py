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
import torch
import timeit
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

    Do we want to load keys from cache file? With sizes of benchmark databases this
    may not be requried. What is more important is that if we get keys from database
    each time we sort of force DB to be cached.

    This Dataset must not be used for real training (though it can easily be adjusted
    for it - see comments in code below - you will to comment 4 lines).
      ~ throughput (images/sec) assuming dataset is cached
      |------------------------------------------|
      | Batch |     pytorch.num_loader_threads   |
      |       | 1    2    4     8     16    32   |
      |------------------------------------------|
      | 32    | 290  558  1103  2080  2865  3635 |
      | 64    | 284  599  1132  2236  3725  4519 |
      | 128   | 298  538  1087  2255  4145  5017 |
      | 256   | 296  562  1108  2330  4648  5385 |
      | 512   | 307  579  1142  2358  4442  6683 |
      | 1024  | 307  554  1125  2283  4605  5974 |
      | 2048  | 315  597  1123  2349  4643  6923 |
      |------------------------------------------|
      It's based on 100 batches with 10 warmup batches on Intel Platinum 8176 @ 2.10GHz.
      Not very accurate but provides intuition on numbers.
    """
    def __init__(self, db_path, effective_batch_size, num_total_batches, transform=None):
        if not HAVE_CAFFE_LMDB:
            raise CAFFE_LMDB_EXCEPTION
        # The epoch change is very expensive (see implementation of DataLoader):
        # http://pytorch.org/docs/master/_modules/torch/utils/data/dataloader.html
        # (in particular, study implementation of the iterator there).
        # Usually, benchmark datasets are small (e.g. 100,000 images) and epoch change
        # happens quite often. To avoid that, we will not be doing this by emulating the
        # presence of large enough dataset.
        # 200 here is just a buffer constant.
        # Comment the following two lines below to use in real training
        print("[WARNING] ***** CaffeLMDBDataset: do not use me in real training *****")
        self.virtual_length = effective_batch_size * num_total_batches + 200
        #
        self.db_path = db_path
        self.env = lmdb.open(db_path, max_readers=126, readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']
        cache_file = os.path.join(db_path, '_cache_')
        if os.path.isfile(cache_file):
            print("[INFO] Loading LMDB keys from cache file (%s)" % (cache_file))
            self.keys = pickle.load(open(cache_file, "rb"))
        else:
            start = timeit.default_timer()
            with self.env.begin(write=False) as txn:
                self.keys = [key for key, _ in txn.cursor()]
            print("[INFO] LMDB database was scanned for keys and it took %f seconds." % (timeit.default_timer() - start))
            print("[INFO] Number of keys = %d" % len(self.keys))
            pickle.dump(self.keys, open(cache_file, "wb"))
        self.transform = transform

    def __getitem__(self, index):
        """
        TODO: This probably needs to be optimized.
        """
        # This is a hack to pretend we have a larger dataset. See constructor comments.
        # Comment the following line to use it in real training.
        index = index % self.length
        #
        # If DB is cached, this is super fast.
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
        # we are emulating larger dataset. See constructor comments.
        # Uncomment the following line to use in real training
        #return self.length
        return self.virtual_length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


class SyntheticDataset(data.Dataset):
    """An implementation of a synthetic dataset. Provides random batch data and labels.

    *** THIS IS NOT USED NOW *** My original idea was to couple this dataset with
    DataLoader. But this seems to work real slow. See below SyntheticDataLoader that's
    used to provide synthetic data.
    """
    def __init__(self, size=1024*200, shape=(3, 227, 227), num_classes=1000):
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


# http://pytorch.org/docs/master/_modules/torch/utils/data/dataloader.html
# It seems that default DataLoader is not efficient or I did not
# figure out how to use it properly with synthetic data.
# This synthetic data loader will iterate forever.
# TODO: Place tensors into GPU memory. This will require rewriting 
#       benchmarking loop and nn.DataParallelModel.
class SyntheticDataLoader(object):
    def __init__(self, batch_size, input_shape, num_classes, device='gpu'):
        # Create two random tensors - one for data and one for label
        data_shape = (batch_size,) + input_shape
        self.data = torch.randn(data_shape)
        self.labels = torch.from_numpy(np.random.randint(0, num_classes, batch_size).astype(np.long))
        if device == 'gpu':
            self.data = self.data.pin_memory()
            self.labels = self.labels.pin_memory()

    def __iter__(self):
        return self

    def next(self):
        return (self.data, self.labels)


class DatasetFactory(object):
    """Creates various dataset loaders"""

    @staticmethod
    def get_data_loader(opts, effective_batch_size, input_shape, num_classes):
        """Creates synthetic/real data loader.

        :param dict opts: A dictionary of input parameters.
        :param int effective_batch_size: Effective batch size. This batch size
                                         will be used by a data loader.
        :param tuple input_shape: A tuple or list of input shape excluding batch
                                  size.
       :param int num_classes: Number of classes.

        :return: An instance of data loader
        """
        if opts['data_dir'] == '':
            #dataset = SyntheticDataset(size=opts['batch_size'] * 200,
            #                           shape=model.module.input_shape,
            #                           num_classes=model.module.num_classes)
            return SyntheticDataLoader(effective_batch_size, input_shape,
                                       num_classes, opts['device'])
        else:
            # Assuming (Channels, Height, Width). This is for image data now.
            # TODO: handle other types of data

            # https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py#L72
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            pipeline = [
                transforms.RandomResizedCrop(input_shape[-1]),
                transforms.ToTensor(),
                normalize,
            ]
            data_backend = opts['data_backend']
            if data_backend == 'image_folder':
                dataset = datasets.ImageFolder(opts['data_dir'], transforms.Compose(pipeline))
            elif data_backend == 'caffe_lmdb':
                dataset = CaffeLMDBDataset(
                    opts['data_dir'],
                    effective_batch_size,
                    opts['num_warmup_batches'] + opts['num_batches'],
                    transforms.Compose(pipeline)
                )
            else:
                raise ValueError("Invalid data backend (%s)" % data_backend)

        #TODO: is here effective batch size used?
        print("[WARNING] DataLoader - check that I accept effective batch size.")
        return data.DataLoader(
            dataset, batch_size=effective_batch_size, shuffle=opts['data_shuffle'],
            num_workers=opts['num_loader_threads'], pin_memory=opts['num_gpus'] >= 1,
            sampler=None
        )
