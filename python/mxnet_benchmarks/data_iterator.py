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
"""Classes defined in this module implement various data iterators.

Data iterators are created after a model has been created, so, shape and layout of input data tensors are known and
cannot be changed.

Each iterator must iterate the following number of times - #warm-up batched + #benchmark batches
DALI was tested to work with mxnet NGC container 19.05.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import mxnet as mx
from mxnet.io import DataBatch, DataIter
import numpy as np
# Try to import horovod
try:
    import horovod.mxnet as hvd
except ImportError:
    hvd = None
# Try to import DALI
try:
    # https://github.com/NVIDIA/DeepLearningExamples/blob/master/MxNet/Classification/RN50v1.5/dali.py
    _mean_pixel = [255 * x for x in (0.485, 0.456, 0.406)]
    _std_pixel = [255 * x for x in (0.229, 0.224, 0.225)]

    from nvidia import dali
    from nvidia.dali.plugin.mxnet import DALIClassificationIterator

    class HybridTrainPipe(dali.pipeline.Pipeline):
        """
        https://docs.nvidia.com/deeplearning/sdk/dali-master-branch-user-guide/docs/api.html
        """
        def __init__(self, batch_size, num_threads, device_id, rec_path, idx_path,
                     shard_id, num_shards, crop_shape,
                     nvjpeg_padding, prefetch_queue=3,
                     output_layout=dali.types.NCHW, pad_output=True, dtype='float16'):
            super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id,
                                                  exec_pipelined=True, prefetch_queue_depth=prefetch_queue)
            self.input = dali.ops.MXNetReader(path=[rec_path], index_path=[idx_path],
                                              random_shuffle=True, shard_id=shard_id, num_shards=num_shards)

            self.decode = dali.ops.nvJPEGDecoder(device="mixed", output_type=dali.types.RGB,
                                                 device_memory_padding=nvjpeg_padding,
                                                 host_memory_padding=nvjpeg_padding)
            self.rrc = dali.ops.RandomResizedCrop(device="gpu", size=crop_shape)
            self.cmnp = dali.ops.CropMirrorNormalize(
                device="gpu",
                output_dtype=dali.types.FLOAT16 if dtype == 'float16' else dali.types.FLOAT,
                output_layout=output_layout,
                crop=crop_shape,
                pad_output=pad_output,
                image_type=dali.types.RGB,
                mean=_mean_pixel,
                std=_std_pixel)
            self.coin = dali.ops.CoinFlip(probability=0.5)
            self.jpegs, self.labels = None, None

        def define_graph(self):
            rng = self.coin()
            self.jpegs, self.labels = self.input(name="Reader")

            images = self.decode(self.jpegs)
            images = self.rrc(images)
            output = self.cmnp(images, mirror=rng)
            return [output, self.labels]
except ImportError:
    dali = None
    DALIClassificationIterator = None


class SyntheticDataIterator(DataIter):
    """ Feeds synthetic (random) data.
    
    See this page for more details:
    https://github.com/apache/incubator-mxnet/blob/master/example/image-classification/common/data.py
    Works with two standard input tensors - data tensor and label tensor.
    """
    def __init__(self, data_shape, label_shape, labels_range, dtype):
        """MXNet partitions data batch evenly among the available GPUs. Here, the
           batch size is the effective batch size.
           
        Memory for data and label tensors is allocated with `cpu_pinned` context.
        To make this iterator consistent with other iterators that provide real data,
        I always set maximal number of iterations to be `warmup_iters + bench_iters`.

        Changes:
            New version does not support `max_iter` parameter. To limit number of batches, wrap this iterator with
                `mx.io.ResizeIter`.

        Args:
            data_shape (tuple): Shape of input data tensor (X) including batch size. The batch size is the 0th
                dimension (bsz = data_shape[0]). This batch size must be an effective batch for a whole node.
            label_shape (tuple): Shape of input label tensor (Y) including batch size. The batch size is the 0th
                dimension (bsz = labels_shape[0]). This batch size must be an effective batch for a whole node.
            labels_range (list): List of output labels. For ImageNet, that would be a list with integers from 0 to 999.
            dtype (str): Data type for data tensor (float32, float16).
        """
        super(SyntheticDataIterator, self).__init__(data_shape[0])
        # Let's assume this data iterator always returns single precision tensors.
        self.dtype = dtype
        # mx.Context: cpu, gpu, cpu_pinned, cpu_shared
        # It seems in latest NGC containers this bug is present: https://github.com/apache/incubator-mxnet/pull/12031
        # The way how mxnet sends data to GPUs, it's better to keep all in CPU. Other possible option would be to
        # eliminate CPU memory at all with synthetic data which is not probably a great idea.
        synth_data_context = os.environ.get('DLBS_MXNET_SYNTHETIC_DATA_CONTEXT', 'cpu')
        self.data = mx.nd.array(np.random.uniform(-1, 1, data_shape),
                                dtype=self.dtype,
                                ctx=mx.Context(synth_data_context, 0))
        self.label_shape = [label_shape[0]]
        if not self.label_shape:
            self.label_shape = [self.batch_size]
        self.label = mx.nd.array(np.random.randint(labels_range[0], labels_range[1] + 1, self.label_shape),
                                 dtype='float32',
                                 ctx=mx.Context(synth_data_context, 0))

    def __iter__(self):
        return self

    @property
    def provide_data(self):
        return [mx.io.DataDesc('data', self.data.shape, self.dtype)]

    @property
    def provide_label(self):
        return [mx.io.DataDesc('softmax_label', self.label_shape, 'float32')]

    def next(self):
        """For DataBatch definition, see this page:
           https://mxnet.incubator.apache.org/api/python/io.html#mxnet.io.DataBatch
        """
        return DataBatch(data=(self.data,), label=(self.label,), pad=0, index=None, provide_data=self.provide_data,
                         provide_label=self.provide_label)

    def __next__(self):
        return self.next()


class DataIteratorFactory(object):
    """A factory that now creates two types of data iterators.
    
    The one is a synthetic data iterator that feeds random tensors, the other one
    is actually an ImageRecordIter.
    """
    @staticmethod
    def get(data_shape, label_shape, labels_range, args, kv_store=None):
        """Creates data iterator.

        Args:
            data_shape (tuple): Shape of input data tensor (X) including batch size. The batch size is the 0th
                dimension (bsz = data_shape[0]). This batch size must be an effective batch for a whole node.
            label_shape (tuple): Shape of input label tensor (Y) including batch size. The batch size is the 0th
                dimension (bsz = labels_shape[0]). This batch size must be an effective batch for a whole node.
            labels_range (list): List of output labels. For ImageNet, that would be a list with integers from 0 to 999.
            args (argparse.Namespace): Command line arguments.
            kv_store (mxnet.kvstore.KVStore): An object returned by mx.kvstore.create('...').

        The data_shape and label_shape have first dimension to be batch dimension. It is a local batch, i.e.:
            replica_batch * num_devices
        Returns:
            Data iterator (instance of mx.io.DataIter).
        """
        logging.info("Creating data iterator: data_shape=%s, label_shape=%s.", data_shape, label_shape)
        # 1. Synthetic Iterator ----------------------------------------------------------------------------------------
        if args.data_dir is None or args.data_dir == "":
            logging.info("Creating synthetic data iterator with data shape = %s.", data_shape)
            return mx.io.ResizeIter(
                SyntheticDataIterator(data_shape, label_shape, labels_range, args.dtype),
                args.num_warmup_batches + args.num_batches
            )
        # 2. Numpy Array Iterator --------------------------------------------------------------------------------------
        fnames = [f for f in os.listdir(args.data_dir) if os.path.isfile(os.path.join(args.data_dir, f))]
        if len(fnames) == 1 and fnames[0].endswith('.npz'):
            dataset = np.load(os.path.join(args.data_dir, fnames[0]))
            data, labels = dataset.get('data', None), dataset.get('labels', None)
            if data is None:
                raise ValueError("The dataset ({}) does not contain 'data' "
                                 "field.".format(os.path.join(args.data_dir, fnames[0])))
            logging.info("Creating NDArray iterator: data=%s, labels=%s", data.shape, labels.shape)
            nd_arr_iter = mx.io.NDArrayIter(data=data, label=labels, batch_size=data_shape[0],
                                            shuffle=False, last_batch_handle='discard')
            return mx.io.ResizeIter(nd_arr_iter, args.num_warmup_batches + args.num_batches)
        # 3. DALI Iterator ---------------------------------------------------------------------------------------------
        if 'horovod' in args.kv_store:
            if not hvd:
                raise ValueError("Horovod library not found")
            rank, nworker = hvd.rank(), hvd.size()
        else:
            rank, nworker = (kv_store.rank, kv_store.num_workers) if kv_store else (0, 1)
        dataset_files = [
            os.path.join(args.data_dir, 'train.rec'),
            os.path.join(args.data_dir, 'train.idx')
        ]
        if os.path.exists(dataset_files[0]) and os.path.exists(dataset_files[1]):
            if args.use_dali is True:
                # https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/examples/mxnet/mxnet-resnet50.html
                if dali is None:
                    raise ValueError("DALI library not found (use_dali is true).")
                if len(args.gpus) == 0:
                    raise ValueError("DALI can only be used with GPU devices (gpus={})".format(args.gpus))
                logging.info("Creating DALI iterator")
                output_layout = dali.types.NHWC if args.input_layout == 'NHWC' else dali.types.NCHW
                cropshape = (data_shape[1], data_shape[2]) if args.input_layout == 'NHWC' else (data_shape[2],
                                                                                                data_shape[3])
                channel_idx = 3 if args.input_layout == 'NHWC' else 1
                trainpipes = [HybridTrainPipe(batch_size=data_shape[0] // len(args.gpus),          # Replica batch.
                                              num_threads=3,                                       # Per GPU
                                              device_id=gpu_id,
                                              rec_path=dataset_files[0],
                                              idx_path=dataset_files[1],
                                              shard_id=args.gpus.index(gpu_id) + len(args.gpus) * rank,
                                              num_shards=len(args.gpus) * nworker,
                                              crop_shape=cropshape,
                                              output_layout=output_layout,
                                              pad_output=data_shape[channel_idx] == 4,
                                              dtype=args.dtype,
                                              nvjpeg_padding=16 * 1024 * 1024,
                                              prefetch_queue=3) for gpu_id in args.gpus]
                trainpipes[0].build()
                # epoch_size = trainpipes[0].epoch_size("Reader") // nworker
                epoch_size = data_shape[0] * (args.num_warmup_batches + args.num_batches)
                return DALIClassificationIterator(
                    trainpipes,                                         # List of pipelines to use
                    epoch_size,                                         # Epoch size.
                    'data',                                             # Data name for provided symbols.
                    'softmax_label',                                    # Label name for provided symbols.
                    args.input_layout                                   # Layout of the pipeline outputs (NCHW / NHWC).
                )

            # 4. MXNET Image Record Iterator ---------------------------------------------------------------------------
            # https://mxnet.incubator.apache.org/api/python/io.html#mxnet.io.imagerecorditer
            # https://github.com/apache/incubator-mxnet/blob/master/example/image-classification/common/data.py
            # this iterator supports channels first format only.
            if args.input_layout != 'NCHW':
                raise ValueError("Standard mxnet image record iterator only supports channel first format (NCHW), "
                                 "requested format: {}.".format(args.input_layout))

            logging.info("Creating standard image record iterator (ImageRecordIter) with data layout = %s.",
                         args.input_layout)
            num_preprocess_threads = args.preprocess_threads
            if num_preprocess_threads <= 4:
                logging.warning("[Number of pre-process threads is %d. This may be too small for large number of GPUs. "
                                "If you do not see speedup as you add more GPUs, increase this number.",
                                num_preprocess_threads)
            img_rec_iter = mx.io.ImageRecordIter(
                path_imgrec=dataset_files[0],
                path_imgidx=dataset_files[1],
                data_name='data',
                label_name='softmax_label',
                data_shape=(data_shape[1], data_shape[2], data_shape[3]),
                batch_size=data_shape[0],
                rand_crop=True,
                rand_mirror=True,
                preprocess_threads=num_preprocess_threads,
                prefetch_buffer=args.prefetch_buffer,
                dtype='float32',
                num_parts=nworker,
                part_index=rank
            )
            return mx.io.ResizeIter(img_rec_iter, args.num_warmup_batches + args.num_batches)

        # 5. All Failed ------------------------------------------------------------------------------------------------
        raise ValueError(
            "Cannot find data set files. MXNET benchmark backend supports the following data sets:\n"
            "  1. Synthetic data set. It is used when data_dir parameter is none or empty:\n"
            "     -Pexp.data_dir='\"\"'\n"
            "  2. Real data set in a file with 'npz' extension. This data set is used if data_dir value\n"
            "     is a valid directory and contains one file with npz extension. If found, this file\n"
            "     must contain a dictionary with at least one key - `data`. It can also contain 'labels'\n"
            "     key for labels.\n"
            "  3. Real image data set in standard RecordIO format. This data set is used if provided data directory\n"
            "     contains 'train.rec' and 'train.idx' files.'"
        )
