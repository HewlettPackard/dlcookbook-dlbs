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
"""
DeepSpeech2 architecture defines a family of neural network models that are
particularly suited for processing speech data. The architecture includes
one or several convolutional layers, one or several recurrent layers, fully
connected layer with softmax and CTC loss function.
Original paper: https://arxiv.org/pdf/1512.02595.pdf

Number of features (nfeatures) - 161, second dim below is feature dim, the first
one is a time dimension.
-------------------------- Convolutional layers -------------------------
conv_arch      channels      filters                      strides             padding
1-layer-1D     1280          (11,nfeatures)               (2,1)]              (0,0)
2-layer-1D     640,640       (5,nfeatures),(5,1)          (1,1),(2,1)         (0,0),(0,0)
3-layer-1D     512,512,512   (5,nfeatures),(5,1),(5,1)    (1,1),(1,1),(2,1)   (0,0),(0,0),(0,0)
1-layer-2D     32            (11,41)                      (2,2)]              (0,0)
2-layer-2D     32,32         (11,41),(11,21)              (2,2),(1,2)         (0,0),(0,0)
2-layer-2D-v2  32,32         (11,41),(11,21)              (3,2),(1,2)         (5,20),(5,10)
3-layer-2D     32,32,96      (11,41),(11,21),(11,21)      (2,2),(1,2),(1,2)   (0,0),(0,0),(0,0)


---------------------------- Recurrent layers ---------------------------
Number of layers: 1, 3, 5 or 7
RNN types:        'rnn_relu', 'rnn_tanh', 'lstm', 'gru'
Bidirectional:    False/True


----------------------------- Important notes ---------------------------
  1. Padding in convolutional layers to keep dimensions. Paper says they use
     'SAME' convolution and then reduce number of time steps they apply RNN to.
  2. Number of classes. If unigram model is used, ~29 classes else ~ 841.
Confirmed:
  1. FusedRNNCell concats outputs from forward/backward branchers. DeepSpeech2
     sums them instead. This script can do both depending on input parameters.
  2. Apply batch norm in each RNN after input transformation. We do not do that.
  3. Optionally share input2hidden weights for BRNN. We do not do this.

In summary (comparing to https://github.com/PaddlePaddle/DeepSpeech):
  1. We can have different conv layers.
  2. RNN layers are different:
      No batch norm.
      No input2hidden weights sharing for BRNN.
      Can have RNN types with dropout.

"""
from __future__ import absolute_import
from __future__ import print_function
import os
import math
import logging
import itertools
import numpy as np
import mxnet as mx
from mxnet_benchmarks.models.model import Model
from mxnet_benchmarks.contrib.ctc_metrics import CtcMetrics

class DeepSpeech2(Model):
    """ Based on MXNET implementation
        https://github.com/apache/incubator-mxnet/blob/master/example/speech_recognition
        https://github.com/apache/incubator-mxnet/blob/master/example/ctc
    Number of features:  161
    Buckets = [200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600]
    """

    implements = 'deep_speech2'

    CONV_ARCHS = [
        '1-layer-1D', '2-layer-1D', '3-layer-1D',    # From paper
        '1-layer-2D', '2-layer-2D', '3-layer-2D',    # From paper
        '2-layer-2D-v2'                              # PaddlePaddle implementation
    ]
    RNN_TYPES = ['rnn_relu', 'rnn_tanh', 'lstm', 'gru']
    BRNN_OUTPUT = ['concat', 'sum']                  # Concat is faster, but DS2 uses 'sum'.
    CTC_LOSSES = ['mxnet_ctc_loss', 'warp_ctc_loss']

    @property
    def output(self):
        return self.__output

    @property
    def batch_size(self):
        """"""
        return self.__batch_size

    @property
    def output_length(self):
        """"""
        return self.__output_length

    def __init__(self, params):
        # Common parameters for all models
        Model.check_parameters(
            params,
            {
                'name': 'DeepSpeech2', 'input_shape':(1, 200, 161),
                'num_classes': 29*29 + 1,  # Alphabet size + BLANK character (which is 0)
                'phase': 'training', 'dtype': 'float32', 'model_opts': {}
            }
        )
        # Specific parameters for DeepSpeech2
        Model.check_parameters(
            params['model_opts'],
            {
                'conv_batch_norm': True,            # Use BatchNorm in Conv layers
                'conv_arch': '2-layer-2D-v2',       # Conv layers architecture
                'num_rnn_layers': 3,                # Number of RNN layers
                'rnn_layer_size': 2048,             # Hidden size of each RNN
                'bidirectional': True,              # Use bidirectional RNN
                'rnn_type': 'rnn_relu',             # One of RNN_TYPES
                'rnn_batch_norm': False,            # Use Batch norm in RNNs after i2h matrix. DOES NOT WORK NOW.
                'brnn_share_i2h': False,            # Share i2h weights in RNNs. DOES NOT WORK NOW.
                'brnn_output': 'concat',            # Aggregate method for BRNN outputs. One of BRNN_OUTPUT
                'rnn_dropout': 0.0,                 # Use dropout in RNNs. Value from [0, 1).
                'ctc_loss': 'mxnet_ctc_loss'        # CTC Loss implementation, one of CTC_LOSSES
            }
        )
        Model.__init__(self, params)
        self.__batch_size = params['batch_size']
        self.__output_length = 0                                   # [output] Length of output sequence
        self.__data_shape = (self.batch_size,) + self.input_shape  # For debugging purposses
        self.__debug = logging.getLogger().isEnabledFor(logging.DEBUG) or ('DLBS_DEBUG' in os.environ and os.environ['DLBS_DEBUG'] == '1')

        if self.model_opts['conv_arch'] not in DeepSpeech2.CONV_ARCHS:
            raise "Invalid conv arch ('%s'), must be one of '%s'" % (self.model_opts['conv_arch'], str(DeepSpeech2.CONV_ARCHS))
        if self.model_opts['rnn_type'] not in DeepSpeech2.RNN_TYPES:
            raise "Invalid RNN type ('%s'), must be one of '%s'" % (self.model_opts['rnn_type'], str(DeepSpeech2.RNN_TYPES))
        if self.model_opts['brnn_output'] not in DeepSpeech2.BRNN_OUTPUT:
            raise "Invalid BRNN output function ('%s'), must be one of '%s'" % (self.model_opts['brnn_output'], str(DeepSpeech2.BRNN_OUTPUT))
        if self.model_opts['ctc_loss'] not in DeepSpeech2.CTC_LOSSES:
            raise "Invalid ctc loss ('%s'), must be one of '%s'" % (self.model_opts['ctc_loss'], str(DeepSpeech2.CTC_LOSSES))
        if self.model_opts['rnn_batch_norm'] is True:
            self.model_opts['rnn_batch_norm'] = False
            print("[WARNING] Batch norm is not supported in RNNs.")
        if self.model_opts['brnn_share_i2h'] is True:
            self.model_opts['brnn_share_i2h'] = False
            print("[WARNING] Sharing input2hidden weights in BRNNs is not supported.")


        print("Model options: " + str(self.model_opts))
        # This helps debugging shapes
        logging.debug("Batch size: %d", self.batch_size)
        logging.debug("Input length: %d", self.input_shape[1])
        logging.debug("Num input features: %d", self.input_shape[2])
        # Input data v is a spectrogram
        v = self.add_data_node()                                  # [Batch, 1, DatumLen, DatumFeatures]
        self.log_shape("Input shape: %s", v)
        # 1-3 layers of 1D or 2D convolutions
        v, length = self.add_conv_layers(v)                       # [Batch, 1, CnnLen, CnnFeatures]
        # Add RNN layers
        v, nrnn_features = self.add_rnn_layers(v, length)         # [CnnLen, Batch, RnnFeatures]
        # Compute CTC loss
        v = mx.sym.Reshape(data=v, shape=(-1, nrnn_features))     # [CnnLen*Batch, RnnFeatures]
        self.log_shape("FC input shape: %s", v)
        v = mx.sym.FullyConnected(data=v,
                                  num_hidden=self.num_classes)    # [CnnLen*Batch, self.num_classes]
        self.log_shape("FC output shape: %s", v)
        if self.dtype == 'float16':
            print("Casting logits to np.float32")
            v = mx.sym.cast(data=v, dtype=np.float32)
        if self.phase == 'training':
            v_ctc = mx.sym.Reshape(data=v,
                                   shape=(length, self.batch_size, self.num_classes))   # [CnnLen, Batch, NumClasses(alphabet+1)]
            labels = mx.sym.Variable(name="softmax_label",
                                     shape=(self.batch_size, length),
                                     init=mx.init.Zero())
            self.log_shape("CTC input shape: %s", v_ctc)
            if self.model_opts['ctc_loss'] == 'warp_ctc_loss':
                print("Using Baidu's Warp CTC Loss.")
                print("[WARNING] WarpCTC was not tested and may not work.")
                try:
                    v = mx.symbol.WarpCTC(data=v_ctc, label=labels)
                except AttributeError:
                    print("[ERROR] WarpCTC symbol is not available. Recompile MXNET with WarpCTC support.")
                    raise
            else:
                print("Using CTCLoss from mx.symbol.contrib.")
                # data:  (sequence_length, batch_size, alphabet_size + 1)
                #        The 0th element of this vector is reserved for the special blank character.
                # label: (batch_size, label_sequence_length)
                #        Is a tensor of integers between 1 and alphabet_size.
                # out:   (batch_size)
                #        Is a list of CTC loss values, one per example in the batch.
                ctc_loss = mx.sym.MakeLoss(mx.symbol.contrib.CTCLoss(data=v_ctc, label=labels, name='ctc'))
                predictions = mx.sym.MakeLoss(mx.sym.SoftmaxActivation(data=v, name='softmax'))
                v = mx.sym.Group([mx.sym.BlockGrad(predictions), ctc_loss])
        else:
            v = mx.symbol.softmax(data=v, name='softmax')
        self.log_shape("Output shape: %s", v)

        self.__output = v
        self.__output_length = length                                    # We have this many labels per input sequence.
        self._labels_shape = (self.__output_length, )                    # K labels for every batch
        self._labels_range = (1, self.num_classes)                       # The class '0' is reserved for BLANK character.

        self.__ctc_metrics = CtcMetrics(seq_len=self.__output_length)
        self._eval_metric = mx.metric.CustomMetric(feval=self.__ctc_metrics.accuracy, name='ctc_metric', allow_extra_outputs=True)

    def add_conv_layers(self, v):
        """ Add convolutional layers.

        :param obj v: Input data

        Convolution kernel size: (w,), (h, w) or (d, h, w)
        Convolution stride: (h, w) or (d, h, w)
        """
        length = self.input_shape[1]
        nfeatures = self.input_shape[2]
        defs = {
            '1-layer-1D':    {'channels': [1280], 'filters': [(11, nfeatures)], 'strides': [(2,1)], 'pads': [(0,0)]},
            '2-layer-1D':    {'channels': [640,640], 'filters': [(5, nfeatures),(5, 1)], 'strides': [(1,1),(2,1)], 'pads': [(0,0),(0,0)]},
            '3-layer-1D':    {'channels': [512,512,512],  'filters': [(5, nfeatures),(5, 1),(5, 1)], 'strides': [(1,1),(1,1),(2,1)], 'pads': [(0,0),(0,0),(0,0)]},
            '1-layer-2D':    {'channels': [32], 'filters': [(11, 41)], 'strides': [(2,2)], 'pads': [(0,0)]},
            '2-layer-2D':    {'channels': [32,32], 'filters': [(11, 41),(11,21)], 'strides': [(2,2),(1,2)], 'pads': [(0,0),(0,0)]},
            '3-layer-2D':    {'channels': [32,32,96], 'filters': [(11,41),(11,21),(11,21)], 'strides': [(2,2),(1,2),(1,2)], 'pads': [(0,0),(0,0),(0,0)]},
            '2-layer-2D-v2': {'channels':  [32,32], 'filters': [(11,41),(11,21)], 'strides': [(3,2),(1,2)], 'pads': [(5,20),(5,10)]}
            # To increase # conv layers in '2-layer-2D-v2' config, replicate parameters of last layer (https://github.com/PaddlePaddle/DeepSpeech/blob/develop/model_utils/network.py).
        }
        arch = defs[self.model_opts['conv_arch']]
        for i in range(len(arch['filters'])):
            name = 'conv%d' % i
            v = DeepSpeech2.conv_bn(
                name, 
                v,
                kernel=arch['filters'][i],
                stride=arch['strides'][i],
                num_channels_out=arch['channels'][i],
                pad=arch['pads'][i],
                batch_norm=self.model_opts['conv_batch_norm'])
            length = int(math.floor((length + 2*arch['pads'][i][0] - arch['filters'][i][0])/arch['strides'][i][0])) + 1
            self.log_shape("Conv '" + name + "' output shape: %s", v)
        logging.debug("Utterance length after conv layers is %d", length)
        return (v, length)

    def add_rnn_layers(self, v, length):
        """Add RNN layers
        """
        # https://mxnet.incubator.apache.org/_modules/mxnet/rnn/rnn_cell.html#FusedRNNCell
        def _begin_state(rnn_cell, func=mx.sym.zeros, **kwargs):
            if self.dtype == 'float32':
                return None   # mxnet will initialzie this
            assert not rnn_cell._modified, \
                "After applying modifier cells (e.g. DropoutCell) the base " \
                "cell cannot be called directly. Call the modifier cell instead."
            states = []
            for info in rnn_cell.state_info:
                rnn_cell._init_counter += 1
                if info is None:
                    state = func(name='%sbegin_state_%d'%(rnn_cell._prefix, rnn_cell._init_counter),dtype=np.float16, **kwargs)
                else:
                    kwargs.update(info)
                    state = func(name='%sbegin_state_%d'%(rnn_cell._prefix, rnn_cell._init_counter),dtype=np.float16, **kwargs)
                states.append(state)
            return states
        #
        rnn_cell = mx.rnn.FusedRNNCell(
            num_hidden=self.model_opts['rnn_layer_size'],
            num_layers=self.model_opts['num_rnn_layers'],
            bidirectional = self.model_opts['bidirectional'],
            mode= self.model_opts['rnn_type'],
            prefix='rnn',
            dropout=self.model_opts['rnn_dropout']
        )
        # Shape of 'v' is [Batch, Channels, Time, Features]. We need to convert it to
        # 'TNC' layout (Time, BatchSize,...)
        v = mx.sym.Reshape(data=v, shape=(length, self.batch_size, -1))
        self.log_shape("RNN input shape: %s", v)
        v,_ = rnn_cell.unroll(
            length=length,
            inputs=v,
            begin_state=_begin_state(rnn_cell),
            layout='TNC',
            merge_outputs=True
        )
        self.log_shape("RNN output shape: %s", v)
        
        nfeatures = self.model_opts['rnn_layer_size']                      # Number of output features. In case of BRNN,
        if self.model_opts['bidirectional']:                               # mnet concats outputs of forward/backward passes.
            if self.model_opts['brnn_output'] == 'sum':                    # DS2 uses sum instead.
                outputs = mx.sym.split(data=v, num_outputs=2, axis=2)   
                v = outputs[0] + outputs[1]
            else:
                nfeatures = nfeatures * 2                                # If not sum, num features for BRNN is doubled.
        return v, nfeatures

    @staticmethod
    def conv_bn(name, input, kernel, stride, num_channels_out, pad=(0,0), batch_norm=False, activation='relu'):
        logging.debug("Adding convolution layer kernel=%s, stride=%s, num_filters=%d, padding=%s",
                      str(kernel), str(stride), num_channels_out,str(pad))
        v = mx.symbol.Convolution(name=name+'_conv', data=input, kernel=kernel,
                                  stride=stride, num_filter=num_channels_out,
                                  no_bias=batch_norm==True, pad=pad)
        if batch_norm:
            logging.debug("Adding batch norm layer")
            v = mx.sym.BatchNorm(name=name+"_bn", data=v, fix_gamma=False,
                                 eps=2e-5, momentum=0.9)
        if activation:
            logging.debug("Adding activation layer '%s'", activation)
            v = mx.symbol.Activation(name=name+'_act', data=v, act_type=activation)
        return v
    
    def shape(self, v):
        """ Return shape of v's output tensor."""
        return str(v.infer_shape(data=self.__data_shape)[1])

    def log_shape(self, pattern, v):
        """ Log shape of 'v' using string 'pattern' if log level is DEBUG.
        """
        if self.__debug:
            logging.debug(pattern, self.shape(v))

    @staticmethod
    def test_configurations():
        """Test different configurations.
        They run in separate process to make sure GPU memory is clean
        before test runs.
        """
        from multiprocessing import Process, Queue
        import sys

        def __worker(q, conv_arch, num_rnn_layers, rnn_layer_size,
                     bidirectional, rnn_type, brnn_output):
            # Uncomment the following two lines if you need MXNET's output
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')
            os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
            device = mx.gpu(0)    
            m = DeepSpeech2({
                'batch_size': 16,
                'model_opts': {
                    'conv_arch': conv_arch,
                    'num_rnn_layers': num_rnn_layers,
                    'rnn_layer_size': rnn_layer_size,
                    'bidirectional': bidirectional,
                    'rnn_type': rnn_type,
                    'brnn_output': brnn_output
                }
            })
            data_shape = (m.batch_size,) + m.input_shape
            data = SyntheticDataIterator(
                m.num_classes, data_shape, max_iter=10, dtype=np.float32,
                label_shape=(m.batch_size, m.output_length)
            )
            mod = mx.mod.Module(symbol=m.output, context=device, label_names=['softmax_label'])
            mod.bind(data_shapes=data.provide_data, label_shapes=data.provide_label, for_training=True, inputs_need_grad=False)
            mod.init_params(initializer=mx.init.Xavier(magnitude=2.))
            mod.init_optimizer(kvstore='local', optimizer='sgd', optimizer_params=(('learning_rate', 0.01),))  
            
            batch = next(data)
            mod.forward_backward(batch)
            mod.update()
            mx.nd.waitall()
            
            q.put(Model.num_parameters(mod))

        params = [
            DeepSpeech2.CONV_ARCHS,    # Conv layers
            [1, 3, 5, 7],              # Number of RNN layers
            [1280],                    # RNN hidden size  2400, 1880, 1510, 1280
            [True, False],             # Bidirectional RNN
            DeepSpeech2.RNN_TYPES,     # Differnt RNN cells
        ]
        
        queue = Queue()
        print("conv_arch rnn_layers rnn_size bidirectional rnn_type [brnn_output]")
        for c in itertools.product(*params):
            if c[3] is False:
                p = Process(target=__worker, args=(queue, c[0], c[1], c[2], c[3], c[4], 'concat'))
                p.start()
                p.join()
                print("%s %d %d %s %s %s" % (c[0], c[1], c[2], 'true' if c[3] else 'false', c[4], queue.get()))
            else:
                for brrn_output in DeepSpeech2.BRNN_OUTPUT: 
                    p = Process(target=__worker, args=(queue, c[0], c[1], c[2], c[3], c[4], brrn_output))
                    p.start()
                    p.join()
                    print("%s %d %d %s %s %s %s" % (c[0], c[1], c[2], 'true' if c[3] else 'false', c[4], brrn_output, queue.get()))  
            


if __name__ == '__main__':
    from mxnet_benchmarks.data_iterator import SyntheticDataIterator
    os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
    #DeepSpeech2.test_configurations()
    logging.getLogger().setLevel(logging.DEBUG)
    
    model = DeepSpeech2({
        'batch_size': 16,
        'dtype': 'float32',
        'model_opts': {
            'conv_arch': '3-layer-1D',
            'num_rnn_layers': 2
        }
    })

    #model.render_to_file(model.output, bsize=model.batch_size, fname='deepspeech2_graph')
    #exit(0)
    
    data_shape = (model.batch_size,) + model.input_shape
    labels_shape = (model.batch_size,) + model.labels_shape
    device = mx.gpu(0)

    data = SyntheticDataIterator(data_shape, labels_shape, model.labels_range,
                                 max_iter=10, dtype=np.float32)

    mod = mx.mod.Module(symbol=model.output, context=device, label_names=['softmax_label'])
    
    mod.bind(data_shapes=data.provide_data, label_shapes=data.provide_label, for_training=True, inputs_need_grad=False)
    mod.init_params(initializer=mx.init.Xavier(magnitude=2.))
    mod.init_optimizer(kvstore='local', optimizer='sgd', optimizer_params=(('learning_rate', 0.01),))    
    
    batch = next(data)
    mod.forward_backward(batch)
    mod.update()
    mx.nd.waitall()

    #print (Model.num_parameters(mod))
    #print (Model.print_parameters(mod))
