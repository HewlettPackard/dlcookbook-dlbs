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

Some of the functionality like fast_collate is from NVIDIA's scripts.
"""
from __future__ import absolute_import
from __future__ import print_function
import os
import pickle
import timeit
import numpy as np
import torch
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


# Disable for entire file "Too few public methods warning"
# pylint: disable=R0903


class CaffeLMDBDataset(torch.utils.data.Dataset):
    """
    https://github.com/pytorch/vision/blob/master/torchvision/datasets/lsun.py#L14
    https://developers.google.com/protocol-buffers/docs/pythontutorial
    https://stackoverflow.com/questions/33117607/caffe-reading-lmdb-from-python
    https://github.com/BVLC/caffe/blob/master/python/caffe/io.py

    Do we want to load keys from cache file? With sizes of benchmark databases this
    may not be requried. What is more important is that if we get keys from database
    each time we sort of force DB to be cached.

    This Dataset must not be used for real training (though it can easily be adjusted
    for it - see comments in code below - you will need to comment 4 lines).
      ~ throughput (images/sec) assuming dataset is cached (in memory)
      |------------------------------------------|
      | Batch |     pytorch.num_loader_threads   |
      |       | 1    2    4     8     16    32   |
      |------------------------------------------|
      | 32    | 347  654  1238  2512  4210  4245 |
      | 64    | 341  675  1282  2495  4712  5845 |
      | 128   | 342  670  1269  2639  5014  5364 |
      | 256   | 345  672  1269  2613  5188  6214 |
      | 512   | 346  678  1261  2518  5169  6388 |
      | 1024  | 351  680  1296  2599  5245  6369 |
      | 2048  | 349  684  1286  2583  5325  4031 |
      |------------------------------------------|
      It's based on 100 batches with 10 warmup batches on Intel Xeon E5-2650 v2 @ 2.30GHz.
      20 cores, 40 with HT (was enabled).
      Not very accurate but provides intuition on numbers.
    """
    def __init__(self, db_path, batch_size, num_total_batches, transform=None):
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
        self.virtual_length = batch_size * num_total_batches + 200
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
            print("[INFO] LMDB database was scanned for keys and it took %f seconds." % \
                  (timeit.default_timer() - start))
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
        # PyLint suggests using if datum.data:, but with len it's more clear I think.
        # pylint: disable=C1801
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
        """Return number of examples in a dataset.

        We are emulating larger dataset. See constructor comments. Uncomment the
        following line to use in real training.
        """
        #return self.length
        return self.virtual_length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


class SyntheticDataLoader(object):
    """ Data loader for synthetic data.

    http://pytorch.org/docs/master/_modules/torch/utils/data/dataloader.html
    It seems that default DataLoader is not efficient or I did not figure out how
    to use it properly with synthetic data. This synthetic data loader will iterate
    forever.
    """
    def __init__(self, opts, input_shape, num_classes):
        """Constructor.
        Args:
            opts: `dict`, Dictionary of options. Must contain `batch_size`, `device`
                          and `data`.
            input_shape: `tuple`, A tuple of input shape of one example (without
                                  batch dimension).
            num_classes: `int`, Number of output classes.
        """
        # Create two random tensors - one for data and one for label
        data_shape = (opts['batch_size'],) + input_shape
        self.data = torch.randn(data_shape)
        self.labels = torch.from_numpy(
            np.random.randint(0, num_classes, opts['batch_size']).astype(np.int_)
        )
        self.prefetchable = True
        if opts['device'] == 'gpu':
            msg = ""
            if opts['data'] in ('synthetic/device', 'synthetic/gpu'):
                self.prefetchable = False
                self.data = self.data.cuda()
                self.labels = self.labels.cuda()
                if opts['dtype'] == 'float16':
                    self.data = self.data.half()
                    msg = "Synthetic dataset will be in device memory in half precision format."
                else:
                    msg = "Synthetic dataset will be in device memory in single precision format."
            elif opts['data'] in ('synthetic', 'synthetic/pinned'):
                self.data = self.data.pin_memory()
                self.labels = self.labels.pin_memory()
                msg = "Synthetic dataset will be in host pinned memory."
            elif opts['data'] == 'synthetic/pageable':
                msg = "Synthetic dataset will be in host pageable memory."
            else:
                raise ValueError("Invalid data type '%s'" % opts['data'])

            if opts['local_rank'] == 0:
                print(msg)

    def next(self):
        """Return next tuple training tuple.
        Returns:
            A tuple of (X, Y)
        """
        return (self.data, self.labels)

    def __iter__(self):
        return self

    def __next__(self):
        return (self.data, self.labels)


def fast_collate(batch):
    """Convert batch into tuple of X and Y tensors."""
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    width = imgs[0].size[0]
    height = imgs[0].size[1]
    tensor = torch.zeros((len(imgs), 3, height, width), dtype=torch.uint8)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if nump_array.ndim < 3:
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)
        tensor[i] += torch.from_numpy(nump_array)
    return (tensor, targets)


class DatasetFactory(object):
    """Creates various dataset loaders"""

    @staticmethod
    def get_data_loader(opts, input_shape, num_classes):
        """Creates synthetic/real data loader.

        Args:
            opts: `dict`, Dictionary of options. Must contain `batch_size`, `device`
                          and `data`.
            input_shape: `tuple`, A tuple of input shape of one example (without
                                  batch dimension).
            num_classes: `int`, Number of output classes.

        Returns:
            An instance (`iterable`) of a data loader.
        """
        if opts['data_dir'] == '':
            return SyntheticDataLoader(opts, input_shape, num_classes)
        else:
            # Assuming (Channels, Height, Width). This is for image data now.
            # TODO: handle other types of data

            # https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py#L72
            # normalize = transforms.Normalize(
            #    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            #)
            pipeline = [
                transforms.RandomResizedCrop(input_shape[-1]),
                transforms.RandomHorizontalFlip()
                #transforms.ToTensor(),
                #normalize,
            ]
            if opts['data_backend'] == 'image_folder':
                dataset = datasets.ImageFolder(opts['data_dir'], transforms.Compose(pipeline))
            elif opts['data_backend'] == 'caffe_lmdb':
                dataset = CaffeLMDBDataset(
                    opts['data_dir'],
                    opts['batch_size'],
                    opts['num_warmup_batches'] + opts['num_batches'],
                    transforms.Compose(pipeline)
                )
            else:
                raise ValueError("Invalid data backend (%s)" % opts['data_backend'])

        if opts['world_size'] > 1:
            dataset_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            dataset_sampler = None
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=opts['batch_size'],
            shuffle=dataset_sampler is None and opts['data_shuffle'],
            num_workers=opts['num_loader_threads'],
            pin_memory=opts['device'] == 'gpu',
            sampler=dataset_sampler,
            collate_fn=fast_collate
        )


class DataPrefetcher(object):
    """Class that prefetches outputs of a data loader into a GPU memory.

    Prefetching is done for GPU devices. Data loader may provide `prefetchable`
    boolean property that additionally controls if a data needs to be prefetched.
    """

    def __init__(self, loader, opts):
        """Constructor.
        Args:
            loader: `iterable`, A data loader that returns (X, Y) tuple. A loader can
                                provide boolean `prefetchable` property to instruct data
                                prefetcher not to prefetch data.
            opts:   `dict`, options that must contain the keys: `device` and `dtype`.
        """
        self.loader = iter(loader)
        self.next_data = None
        self.next_labels = None
        self.prefetchable = opts['device'] == 'gpu' and \
                            (not getattr(loader, 'prefetchable', None) or loader.prefetchable)
        if self.prefetchable:
            self.prefetchable = True
            self.fp16 = opts['dtype'] == 'float16'
            self.stream = torch.cuda.Stream()
            self.__preload()

    def next(self):
        """Return next tuple (X,Y) that can be prefetched into GPU memory.
        Returns:
            A tuple of (X, Y).
        """
        if not self.prefetchable:
            self.__get_next()
            return (self.next_data, self.next_labels)
        if self.next_data is None:
            return (None, None)
        if not self.next_data.is_cuda:
            torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        labels = self.next_labels
        self.__preload()
        return (data, labels)

    def __get_next(self):
        try:
            self.next_data, self.next_labels = next(self.loader)
        except StopIteration:
            self.next_data = None
            self.next_labels = None

    def __preload(self):
        self.__get_next()
        if self.next_data is None or self.next_data.is_cuda:
            return
        with torch.cuda.stream(self.stream):
            self.next_data = self.next_data.cuda(async=True)
            self.next_labels = self.next_labels.cuda(async=True)
            if self.fp16:
                self.next_data = self.next_data.half()
            else:
                self.next_data = self.next_data.float()
