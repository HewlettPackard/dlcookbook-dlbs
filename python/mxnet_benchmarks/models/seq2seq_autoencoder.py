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


class DeviceMSE(mx.metric.EvalMetric):
    """Mean-Squared Error metric that computes metrics on GPU."""
    def __init__(self, name='mse', output_names=None, label_names=None):
        super(DeviceMSE, self).__init__(name, output_names=output_names, label_names=label_names)

    def update(self, labels, preds):
        #print (labels.shape)
        #print (preds.shape)
        if not isinstance(labels, list):
            labels = [labels]
        if not isinstance(preds, list):
            preds = [preds]
        num_symbols = len(labels)
        if num_symbols != len(preds):
            raise "[ERROR] DeviceMSE::update requries equal number of label nad prediction "\
                  "tensors. Got %d label tensors and %d prediction tensors" % (len(labels), len(preds))
        for i in range(num_symbols):
            # Labels and predictions may have different context
            label = labels[i]
            pred = preds[i]
            if label.context != pred.context:
                pred = pred.as_in_context(label.context)
            self.sum_metric = ((label-pred)**2.0).mean().asscalar()
            self.num_inst += 1


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
                print("Adding random Gaussian noise to input data.")
                noise = mx.sym.random_normal(loc=self.model_opts['transform_mean'],
                                             scale=self.model_opts['transform_stdev'],
                                             dtype=self.dtype)
                v = inputs + noise
            elif self.model_opts['transform'] == 'dropout':
                print("Setting random data elements to zero.")
                mask = mx.sym.random_uniform(0.0, 1.0, dtype=self.dtype) < self.model_opts['transform_drop_prob']
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
        #self._eval_metric = 'mse'
        self._eval_metric = DeviceMSE()

        length = self.input_shape[0]      # Length of a time series
        nfeatures = self.input_shape[1]   # Number of features
        # Input data
        #data, data32 = self.add_data_node()                                      # data: [Batch, Length, Features=50]
        data = mx.sym.Variable(name='data', dtype=np.float32)
        if self.dtype == 'float16':
            print("Casting input DATA tensor to np.float16")
            v = mx.sym.cast(data=data, dtype=np.float16)
        else:
            v = data
        #print("Input data type: " + str(v.infer_type(data='float32')))
        # Apply input function (corrupt data)
        #v = self.add_noise(data)                                         # v:    [Batch, Length, Features=50]
        v = self.add_noise(v)                                         # v:    [Batch, Length, Features=50]
        #print("Noisy output type: " + str(v.infer_type(data='float32')))
        # We need to convert it to RNN's 'TNC' layout
        v = mx.sym.Reshape(data=v, shape=(length, -1, nfeatures))        # v:    [Length, Batch, Features=50]
        #print("Encoder input type: " + str(v.infer_type(data='float32')))
        # RNN layers in encoder/decoder won't change sequence length.
        # Add encoder. One bidirectional LSTM layer followed by 2 
        #              unidirectional LSTM layers
        common_opts = {'hidden_size': 128, 'rnn_type': 'lstm'}
        for i, opts in enumerate([{'bidirectional': True, 'brnn_output': 'concat'}, {}, {'return_state': True, 'merge_outputs': False}]):
            layer_opts = common_opts.copy()
            layer_opts.update(opts)
            v, states, num_features = self.add_rnn_layers(               # v: [Length, Batch, HiddenSize=128]
                v, length, layer_opts, 'encoder_lstm_%d' % (i+1)
            )
            if not isinstance(v, list):
                print (str(v.list_arguments()))
                print("Encoder layer %d output type: %s" % (i+1,str(v.infer_type(data='float32'))))
            else:
                #print (str(v[-1].list_arguments()))
                #print("Encoder layer %d output type: %s" % (i+1,str(v[-1].infer_type(data='float32'))))
                pass
            #print (len(states))
            #if len(states) > 0:
            #    print ("Infering state shapes")
            #    print("Encoder output state 0 type: %s" % (str(states[0].infer_type(data='float32'))))
            #    print("Encoder output state 1 type: %s" % (str(states[1].infer_type(data='float32'))))
        # Take states for layer 0
        states = (
            mx.sym.Reshape(states[0], shape=(-1, num_features)),
            mx.sym.Reshape(states[1], shape=(-1, num_features))
        )
        #print("Decoder initial state 0 type: %s" % (str(states[0].infer_type(data='float16'))))
        #print("Decider initial state 1 type: %s" % (str(states[1].infer_type(data='float16'))))
        # Decoder. We need to manually unroll the very first layer, next layers can be added automatically.
        # 1. Initial states for this LSTM are the states from last encoder LSTM
        decoder_cell = mx.rnn.LSTMCell(num_hidden=128, prefix='decoder_lstm_1')
        # Outputs at every time step of the first decoder layer
        decoder_outputs = [None] * length
        # Output of the last encoder layer
        v = v[-1]                                                        # v: [Batch, HiddenSize=128]
        #print("Decoder input type: %s" % (str(v.infer_type(data='float16')[1])))
        for step in range(length):
            v, states = decoder_cell(v, states)
            #print("Decoder output type: %s" % (str(v.infer_type(data='float32')[1])))
            decoder_outputs[step] = v
        v = decoder_outputs
        # 2. Add second and thrid LSTM layers
        for i in [2, 3]:
            v, _, num_rnn_features = self.add_rnn_layers(                    # v: [Length, Batch, HiddenSize=128]
                v, length, {'hidden_size': 128, 'rnn_type': 'lstm'}, 'decoder_lstm_%d' % i
            )
            #print("Decoder output type: %s" % (str(v.infer_type(data='float16')[1])))
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
        #print("Linear regression input predictions type: %s" % (str(v.infer_type(data='float16')[1])))
        #print("Linear regression input targets type: %s" % (str(data.infer_type(data='float32')[1])))
        if self.phase == 'training':
            #mx.sym.Variable(name="softmax_label", init=mx.init.Zero())
            v = mx.sym.LinearRegressionOutput(data=v, label=data, name='lro')
        self.__output = v


if __name__ == '__main__':
    from mxnet_benchmarks.data_iterator import SyntheticDataIterator
    os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

    batch_size = 16
    model = Seq2SeqAutoencoder({
        'dtype': 'float32',
        'model_opts': {
            'transform': 'identity'
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
    
    #metrics  = mx.metric.MSE()
    metrics  = DeviceMSE()
    for i in range(1):
        batch = next(data)
        #print(batch.data[0][0][0][0:5])
        output = mod.forward(batch)
        #print(mod.get_outputs()[0][0][0][0:5])
        mod.backward()
        mod.update()
        #mod._exec_group.data_arrays
        #     Len = # InputTensors
        # Len(mod._exec_group.data_arrays[i]) = # GPUs
        # mod._exec_group.data_arrays[i][j] - (slice, input_Data)
        #print(mod._exec_group.data_arrays)
        
        #print(mod._exec_group.data_arrays[0][0][1].asnumpy().shape)
        
        for i, texec in enumerate(mod._exec_group.execs):
            #print(texec.outputs[0].asnumpy().shape)
            #print(mod._exec_group.data_arrays[0][i][1].asnumpy().shape)
            l = mod._exec_group.data_arrays[0][i][1]
            p = texec.outputs[0]
            print ("Computing metrics")
            metrics.update(labels=l, preds=p)
            print(metrics.get())
        
        #print(mod._exec_group.execs)
        #print(mod._exec_group.slices)
        #print(batch.data[0].asnumpy().dtype)
        #print (len(mod.get_outputs()))
        #print(mod.get_outputs(merge_multi_context=False)[0][0].asnumpy().dtype)
        #metrics.update(labels=batch.data[0], preds=mod.get_outputs()[0])
        mx.nd.waitall()
    
    #print (Model.print_parameters(mod))
    #model.render_to_file(model.output, 1, "seq2seq_autoencoder")
    print("Done")
