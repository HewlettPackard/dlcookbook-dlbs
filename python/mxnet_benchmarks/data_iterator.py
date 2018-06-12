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
"""Classes defined in this module implement various data iterators."""
import mxnet as mx
from mxnet.io import DataBatch, DataIter
import numpy as np


class SyntheticDataIterator(DataIter):
    """ Feeds synthetic (random) data.

    See this page for more details:
    https://github.com/apache/incubator-mxnet/blob/master/example/image-classification/common/data.py
    Works with two standard input tensors - data tensor and label tensor.
    """
    def __init__(self, data_shape, **kwargs):
        """MXNet partitions data batch evenly among the available GPUs. Here, the
           batch size is the effective batch size.

        Memory for data and label tensors is allocated with `cpu_pinned` context.
        To make this iterator consistent with other iterators that provide real data,
        I always set maximal number of iterations to be `warmup_iters + bench_iters`.

        :param tuple data_shape: Shape of input data tensor (X) including batch size.
                                 The batch size if the 0-th dimension (bsz = data_shape[0])
        :param dict kwargs: Other named arguments:
                  max_iter: Maximal number of iterations to perform. Basically, emulates
                            the dataset size. If negative, will iterate forever and will
                            never throw `StopIteration` exception.
                  dtype: Type of data (float32, float16).
                  provide_labels: If true, data iterator must provide labels (supervised
                                  dataset)
                  label_shape: [only of provide_labels is True]
                  label_range: [only of provide_labels is True]
        """
        super(SyntheticDataIterator, self).__init__(data_shape[0])
        def _set(args, key, value):
            if key not in args: args[key] = value
        _set(kwargs, 'max_iter', 100)
        _set(kwargs, 'dtype', np.float32)
        _set(kwargs, 'provide_labels', True)
        _set(kwargs, 'labels_shape', None)
        _set(kwargs, 'labels_range', None)

        self.cur_iter = 0
        self.max_iter = kwargs['max_iter']
        self.dtype = kwargs['dtype']
        self.data = mx.nd.array(
            np.random.uniform(-1, 1, data_shape),
            dtype=self.dtype,
            ctx=mx.Context('cpu_pinned', 0)
        )
        if not kwargs['provide_labels']:
            self.labels_shape = None
        else:
            self.labels_shape = kwargs['labels_shape']
            if not self.labels_shape:
                self.labels_shape = [self.batch_size,]
            labels_range = kwargs['labels_range']
            self.labels = mx.nd.array(
                np.random.randint(labels_range[0], labels_range[1] + 1, self.labels_shape),
                dtype=self.dtype,
                ctx=mx.Context('cpu_pinned', 0)
            )

    def __iter__(self):
        return self

    @property
    def provide_data(self):
        return [mx.io.DataDesc('data', self.data.shape, self.dtype)]

    @property
    def provide_label(self):
        if not self.labels_shape:
            return None
        else:
            return [mx.io.DataDesc('softmax_label', self.labels_shape, self.dtype)]

    def next(self):
        """For DataBatch definition, see this page:
           https://mxnet.incubator.apache.org/api/python/io.html#mxnet.io.DataBatch
        """
        self.cur_iter += 1
        if self.max_iter < 0 or self.cur_iter <= self.max_iter:
            if self.labels_shape is not None:
                return DataBatch(data=(self.data,), label=(self.labels,),
                                 pad=0, index=None,
                                 provide_data=self.provide_data, provide_label=self.provide_label)
            else:
                return DataBatch(data=(self.data,), pad=0, index=None, provide_data=self.provide_data)
        else:
            raise StopIteration

    def __next__(self):
        return self.next()

    def reset(self):
        self.cur_iter = 0


class DataIteratorFactory(object):
    """A factory that now creates two types of data iterators.

    The one is a synthetic data iterator that feeds random tensors, the other one
    is actually an ImageRecordIter.
    """
    @staticmethod
    def get(data_shape, opts, kv_store=None, **kwargs):
        """Creates data iterator.

        :param int num_classes: Number of classes.
        :param tuple data_shape: Shape of input data tensor (X) including batch size.
                                 The batch size if the 0th dimension (bsz = data_shape[0]).
                                 This batch size must be an affective batch for a whole node.
        :param dict opts: Benchmark parameters
        :param kv_store: An object returned by mx.kvstore.create('...')
        :return: Data iterator.
        """
        data_iter = None
        if 'data_dir' not in opts or not opts['data_dir']:
            data_iter = SyntheticDataIterator(
                data_shape,
                max_iter=opts['num_warmup_batches'] + opts['num_batches'],
                dtype='float32',
                **kwargs
            )
        else:
            if kv_store:
                (rank, nworker) = (kv_store.rank, kv_store.num_workers)
            else:
                (rank, nworker) = (0, 1)
            # https://mxnet.incubator.apache.org/api/python/io.html#mxnet.io.ImageRecordIter
            # https://github.com/apache/incubator-mxnet/blob/master/example/image-classification/common/data.py
            data_iter = mx.io.ImageRecordIter(
                path_imgrec=opts['data_dir'],
                data_name='data',
                label_name='softmax_label',
                data_shape=(data_shape[1], data_shape[2], data_shape[3]),
                batch_size=data_shape[0],
                rand_crop=True,
                rand_mirror=True,
                #dtype=opts['dtype'],
                preprocess_threads=opts.get('preprocess_threads', 4),
                prefetch_buffer=opts.get('prefetch_buffer ', 10),
                dtype='float32',
                num_parts=nworker,
                part_index=rank
            )
        return data_iter
