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
"""
from __future__ import absolute_import
from __future__ import print_function
import os
import mxnet as mx
import numpy as np
from mxnet_benchmarks.models.model import Model


class Seq2SeqAutoencoder(Model):
    implements = 'seq2seq_autoencoder'

    @property
    def output(self):
        return self.__output


    RNN_TYPES = ['rnn_relu', 'rnn_tanh', 'lstm', 'gru']
    BRNN_OUTPUT = ['concat', 'sum']
    TRANSFORM = ['identity', 'gaussian', 'dropout']

    def add_rnn_layers(self, inputs, length, layer_opts, prefix):
        """Add one or several RNN layers defined by a property object 'layer_opts'
        If this is a stack of bidirectional layers and merge operator is 'sum',
        this 'sum' operator will only be applied to the very last layer i.e. to the
        output of this stack of RNN layers. Intermidiate BRNN layers will use 'concat'
        operator. 

        :param obj inputs: Input symbol
        :param int length: Length of a sequence. Will not be changed.
        :param dict layer_opts: Property for stack of RNN layers.
        :param str prefix: A prefix name for RNN layers.
        :return: A tuple of (output symbol, list of hidden states, number of output features)
                 Pay attention that tensors in hidden states will have first dimension being
                 layer dimension.
        """
        if 'merge_outputs' not in layer_opts: layer_opts['merge_outputs'] = True
        if 'return_state' not in layer_opts:  layer_opts['return_state'] = False
        if 'bidirectional' not in layer_opts:  layer_opts['bidirectional'] = False
        if 'num_layers' not in layer_opts:  layer_opts['num_layers'] = 1
            
            
        for opt in ['hidden_size', 'rnn_type', 'num_layers', 'bidirectional', 'return_state', 'merge_outputs']:
            if opt not in layer_opts:
                raise "RNN layer spec (%s) is missing parameter '%s'" % (str(layer_opts), opt)
        if layer_opts['bidirectional'] and 'brnn_output' not in layer_opts:
            raise "RNN layer spec (%s) is missing parameter '%s'" % (str(layer_opts), 'brnn_output')

        rnn_cell = mx.rnn.FusedRNNCell(
            num_hidden=layer_opts['hidden_size'],
            num_layers=layer_opts['num_layers'],
            bidirectional=layer_opts['bidirectional'],
            mode=layer_opts['rnn_type'],
            prefix=prefix,
            dropout=0,
            get_next_state=layer_opts['return_state']
        )
        v, states = rnn_cell.unroll(
            length=length, inputs=inputs,
            begin_state=self.rnn_begin_state(rnn_cell),
            layout='TNC', merge_outputs=layer_opts['merge_outputs']
        )
        num_rnn_features = layer_opts['hidden_size']
        if layer_opts['bidirectional']:
            if layer_opts['brnn_output'] == 'sum':
                outputs = mx.sym.split(data=v, num_outputs=2, axis=2)
                v = outputs[0] + outputs[1]
            else:
                num_rnn_features = 2 * num_rnn_features
        return (v, states, num_rnn_features)

    def add_noise(self, inputs):
        if self.phase == 'training':
            if self.model_opts['transform'] == 'gaussian':
                noise = mx.sym.random_normal(loc=self.model_opts['transform_mean'],
                                             scale=self.model_opts['transform_stdev'],
                                             dtype=self.dtype)
                v = inputs + noise
            elif self.model_opts['transform'] == 'dropout':
                mask = mx.sym.random_uniform(0.0, 1.0) < self.model_opts['transform_drop_prob']
                v = inputs * mask
            else:
                v = inputs
        else:
            v = inputs
        return v

    def __init__(self, params):
        # Common parameters for all models
        Model.check_parameters(
            params,
            {
                'name': 'Seq2SeqAutoencoder',
                'input_shape':(200, 50),  # (Time, Features)
                'num_classes': 50,        # Number of outputs, same as number of inputs
                'phase': 'training', 'dtype': 'float32', 'model_opts': {}
            }
        )
        # Specific parameters for RNN Autoencoder
        Model.check_parameters(
            params['model_opts'],
            {
                'transform': 'identity',            # One of TRANSFORM
                'transform_mean': 0,                # If transform in 'gaussian', this is the mean
                'transform_stdev': 0.001,           # If transform in 'gaussian', this is the standard deviation
                'transform_drop_prob': 0.2,         # If transform in 'fropout', this is the dropout probability
            }
        )
        Model.__init__(self, params)
        self._needs_labels = False
        self._eval_metric = 'mse'

        length = self.input_shape[0]      # Length of a time series
        nfeatures = self.input_shape[1]   # Number of features
        # Input data
        data = self.add_data_node()                                      # data: [Batch, Length, Features=50]
        # Apply input function (corrupt data)
        v = self.add_noise(data)                                         # v:    [Batch, Length, Features=50]
        # We need to convert it to RNN's 'TNC' layout
        v = mx.sym.Reshape(data=v, shape=(length, -1, nfeatures))        # v:    [Length, Batch, Features=50]
        # RNN layers in encoder/decoder won't change sequence length.
        # Add encoder. One bidirectional LSTM layer followed by 2 
        #              unidirectional LSTM layers
        common_opts = {'hidden_size': 128, 'rnn_type': 'lstm'}
        for i, opts in enumerate([{'bidirectional': True, 'brnn_output': 'sum'}, {}, {'return_state': True, 'merge_outputs': False}]):
            layer_opts = common_opts.copy()
            layer_opts.update(opts)
            v, states, num_features = self.add_rnn_layers(               # v: [Length, Batch, HiddenSize=128]
                v, length, layer_opts, 'encoder_lstm_%d' % (i+1)
            )
        # Take states for layer 0
        states = (
            mx.sym.Reshape(states[0], shape=(-1, num_features)),
            mx.sym.Reshape(states[1], shape=(-1, num_features))
        )
        # Decoder. We need to manually unroll the very first layer, next layers can be added automatically.
        # 1. Initial states for this LSTM are the states from last encoder LSTM
        decoder_cell = mx.rnn.LSTMCell(num_hidden=128, prefix='decoder_lstm_1')
        # Outputs at every time step of the first decoder layer
        decoder_outputs = [None] * length
        # Output of the last encoder layer
        v = v[-1]                                                        # v: [Batch, HiddenSize=128]
        for step in range(length):
            v, states = decoder_cell(v, states)
            decoder_outputs[step] = v
        v = decoder_outputs
        # 2. Add second and thrid LSTM layers
        for i in [2, 3]:
            v, _, num_rnn_features = self.add_rnn_layers(                    # v: [Length, Batch, HiddenSize=128]
                v, length, {'hidden_size': 128, 'rnn_type': 'lstm'}, 'decoder_lstm_%d' % i
            )
        # The output of RNN layers is a tensor of shape [Length, Batch, Features]
        v = mx.sym.Reshape(data=v, shape=(-1, num_rnn_features))         # [Length*Batch,  RnnHidden]
        #print("Decoder reshaped output: " + str(v.infer_shape(data=(16,200,50))[1]))
        v = mx.sym.FullyConnected(data=v,
                                  num_hidden=self.num_classes)           # [Length*Batch,  Features]
        # Reshape to original shape
        v = mx.sym.Reshape(data=v, shape=(-1, length, nfeatures))        # [Batch, Length, Features]
        #print("Autoencoder output: " + str(v.infer_shape(data=(16,200,50))[1]))
        if self.dtype == 'float16':
            print("Casting logits to np.float32")
            v = mx.sym.cast(data=v, dtype=np.float32)
        if self.phase == 'training':
            #mx.sym.Variable(name="softmax_label", init=mx.init.Zero())
            v = mx.sym.LinearRegressionOutput(data=v, label=data)
        self.__output = v


if __name__ == '__main__':
    from mxnet_benchmarks.data_iterator import SyntheticDataIterator
    os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

    batch_size = 16
    model = Seq2SeqAutoencoder({
        'dtype': 'float32',
        'model_opts': {
            'transform': 'dropout'
        }
    })

    data_shape = (batch_size,) + model.input_shape
    device = mx.gpu(0)

    data = SyntheticDataIterator(
        data_shape,
        labels_shape=((batch_size,) + model.labels_shape) if model.needs_labels else None,
        labels_range=model.labels_range if model.needs_labels else None,
        max_iter=10,
        dtype=np.float32,
        provide_labels=model.needs_labels
    )
    mod = mx.mod.Module(
        symbol=model.output, context=device,
        label_names=['softmax_label'] if model.needs_labels else None
    )
    mod.bind(
        data_shapes=data.provide_data,
        label_shapes=data.provide_label if model.needs_labels else None,
        for_training=True, inputs_need_grad=False
    )
    mod.init_params(initializer=mx.init.Xavier(magnitude=2.))
    mod.init_optimizer(kvstore='local', optimizer='sgd', optimizer_params=(('learning_rate', 0.01),))
    """
    for i in range(2):
        batch = next(data)
        #print(batch.data[0][0][0][0:5])
        output = mod.forward(batch)
        #print(mod.get_outputs()[0][0][0][0:5])
        mod.backward()
        mod.update()
        mx.nd.waitall()
    """
    print (Model.print_parameters(mod))
    print("Done")
