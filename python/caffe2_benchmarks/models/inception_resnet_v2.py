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
"""Based on mxnet's implementation:
https://github.com/apache/incubator-mxnet/blob/master/example/image-classification/symbols/inception-resnet-v2.py

Visualization (may be slightly different from this implementation):
http://ethereon.github.io/netscope/#/gist/95826c28109d1a5ab95306c37bb30225

This class provides additional functionality not found in other models
that allows integrating this code with resnet50_trainer.py

"""
from __future__ import absolute_import
from caffe2.python import brew
from caffe2_benchmarks.models.model import Model

class InceptionResNetV2(Model):

    implements = 'inception_resnet_v2'

    def __init__(self, params):
        Model.check_parameters(
            params,
            {'name': 'InceptionResNetV2', 'input_shape':(3, 299, 299),
             'num_classes': 1000, 'arg_scope': {'order': 'NCHW'}}
        )
        Model.__init__(self, params)
        self.__run_with_resnet50_trainer = False
        if 'run_with_resnet50_trainer' in params:
            self.__run_with_resnet50_trainer = params['run_with_resnet50_trainer']

    def conv_factory(self, model, v, num_in_channels, num_filters, kernel,
                     stride=1, pad=0, relu=True, name='conv'):
        """Standard convolution block: Conv -> BatchNorm -> Activation
        """
        if isinstance(pad, int):
            pad_t = pad_b = pad_l = pad_r = pad
        elif isinstance(pad, list) or isinstance(pad, tuple):
            if len(pad) == 2:
                pad_t = pad_b = pad[0]
                pad_l = pad_r = pad[1]
            elif len(pad) == 4:
                pad_t = pad[0]
                pad_b = pad[1]
                pad_l = pad[2]
                pad_r = pad[3]
            else:
                assert False, "Invalid length of pad array. Expecting 2 or 4 but have: " + str(pad)
        else:
            assert False, "Invalid type of padding: " + str(pad)

        v = brew.conv(model, v, name + '_conv', num_in_channels, num_filters,
                      kernel=kernel, pad_t=pad_t, pad_l=pad_l, pad_b=pad_b,
                      pad_r=pad_r, stride=stride)
        v = brew.spatial_bn(model, v, name+'_bn', num_filters, eps=2e-5,
                            momentum=0.9, is_test=(self.phase == 'inference'))
        if relu is True:
            v = brew.relu(model, v, name + '_relu')
        return v

    def block_head(self, model, v, towers, num_in_channels, num_out_channels,
                   scale=1.0, relu=True, name='block_head_node'):
        tower_mixed = brew.concat(model, towers, blob_out=name+'_tower_mixed')
        tower_out = self.conv_factory(model, tower_mixed, num_in_channels,
                                      num_filters=num_out_channels,
                                      kernel=1, relu=relu, name=name+'tower_out')
        #v = v + scale * tower_out
        scaled = model.Scale(tower_out, name + '_scale', scale=scale)
        v = brew.sum(model, [v, scaled], name+'_sum')
        #
        if relu is True:
            v = brew.relu(model, v, name + '_relu')
        return v

    def block35(self, model, v, num_in_channels, scale=1.0, relu=True, name='block35'):
        towers = [None, None, None]
        towers[0] = self.conv_factory(model, v, num_in_channels, num_filters=32,
                                      kernel=1, name=name+'tower1_1')
        towers[1] = self.conv_factory(model, v, num_in_channels, num_filters=32,
                                      kernel=1, name=name+'tower2_1')
        towers[1] = self.conv_factory(model, towers[1], 32, num_filters=32,
                                      kernel=3, pad=1, name=name+'tower2_2')
        towers[2] = self.conv_factory(model, v, num_in_channels, num_filters=32,
                                      kernel=1, name=name+'tower3_1')
        towers[2] = self.conv_factory(model, towers[2], 32, num_filters=48,
                                      kernel=3, pad=1, name=name+'tower3_2')
        towers[2] = self.conv_factory(model, towers[2], 48, num_filters=64,
                                      kernel=3, pad=1, name=name+'tower3_3')
        return self.block_head(model, v, towers, num_in_channels=32+32+64,
                               num_out_channels=num_in_channels,
                               scale=scale, relu=relu, name=name)

    def block17(self, model, v, num_in_channels, scale=1.0, relu=True, name='block17'):
        towers = [None, None]
        towers[0] = self.conv_factory(model, v, num_in_channels, num_filters=192,
                                      kernel=1, name=name+'_tower1_1')
        towers[1] = self.conv_factory(model, v, num_in_channels, num_filters=129,
                                      kernel=1, name=name+'tower2_1')
        towers[1] = self.conv_factory(model, towers[1], 129, num_filters=160,
                                      kernel=[1, 7], pad=[1, 2], name=name+'tower2_2')
        towers[1] = self.conv_factory(model, towers[1], 160, num_filters=192,
                                      kernel=[7, 1], pad=[2, 1], name=name+'tower2_3')
        return self.block_head(model, v, towers, num_in_channels=192+192,
                               num_out_channels=num_in_channels,
                               scale=scale, relu=relu, name=name)

    def block8(self, model, v, num_in_channels, scale=1.0, relu=True, name='block8'):
        towers = [None, None]
        towers[0] = self.conv_factory(model, v, num_in_channels, num_filters=192,
                                      kernel=1, name=name+'_tower1_1')
        towers[1] = self.conv_factory(model, v, num_in_channels, num_filters=192,
                                      kernel=1, name=name+'tower2_1')
        towers[1] = self.conv_factory(model, towers[1], 192, num_filters=224,
                                      kernel=[1, 3], pad=[0, 1], name=name+'tower2_2')
        towers[1] = self.conv_factory(model, towers[1], 224, num_filters=256,
                                      kernel=[3, 1], pad=[1, 0], name=name+'tower2_3')
        return self.block_head(model, v, towers, num_in_channels=192+256,
                               num_out_channels=num_in_channels,
                               scale=scale, relu=relu, name=name)


    def forward_pass_builder(self, model, loss_scale=1.0):
        v = 'data' # 3x299x299
        #
        conv1 = self.conv_factory(model, v, 3, num_filters=32, kernel=3, stride=2, name='conv1')   #32x149x149
        conv2 = self.conv_factory(model, conv1, 32, 32, kernel=3, name='conv2')                   #32x147x147
        conv3 = self.conv_factory(model, conv2, 32, 64, kernel=3, pad=1, name='conv3')            #64x147x147
        pool1 = brew.max_pool(model, conv3, 'pool1', kernel=3, stride=2)                          #64x73x73
        #
        conv4r = self.conv_factory(model, pool1, 64, 80, kernel=1, name='conv4_reduce')           #80x73x73
        conv4 = self.conv_factory(model, conv4r, 80, 192, kernel=3, name='conv4')                 #192x71x71
        pool2 = brew.max_pool(model, conv4, 'pool2', kernel=3, stride=2)                          #192x35x35
        #
        conv5 = [None, None, None, None]

        conv5[0] = self.conv_factory(model, pool2, 192, 96, kernel=1, name='conv5_1_1')           #96x35x35

        conv5[1] = self.conv_factory(model, pool2, 192, 48, kernel=1, name='conv5_2_1')           #48x35x35
        conv5[1] = self.conv_factory(model, conv5[1], 48, 64, kernel=5, pad=2, name='conv5_2_2')  #64x35x35

        conv5[2] = self.conv_factory(model, pool2, 192, 64, kernel=1, name='conv5_3_1')           #64x35x35
        conv5[2] = self.conv_factory(model, conv5[2], 64, 96, kernel=3, pad=1, name='conv5_3_2')  #96x35x35
        conv5[2] = self.conv_factory(model, conv5[2], 96, 96, kernel=3, pad=1, name='conv5_3_3')  #96x35x35

        conv5[3] = brew.average_pool(model, pool2, 'conv5_4_1_pool', kernel=3, stride=1, pad=1)   #192x35x35
        conv5[3] = self.conv_factory(model, conv5[3], 192, 64, kernel=1, name='conv5_4_2')        #64x35x35

        conv5 = brew.concat(model, conv5, blob_out='conv5')                                       #320x35x35
        #
        block35 = conv5
        for i in range(10):
            block35 = self.block35(model, block35, num_in_channels=320,
                                   scale=0.17, name='inception_resnet_v2_a%d'%(i+1))              #320x35x35
        # ra - reduction_a
        ra = [None, None, None]

        ra[0] = self.conv_factory(model, block35, 320, 384, kernel=3, stride=2, name='ra_1_1')    #384x17x17

        ra[1] = self.conv_factory(model, block35, 320, 256, kernel=1, name='ra_2_1')              #256x35x35
        ra[1] = self.conv_factory(model, ra[1], 256, 256, kernel=3, pad=1, name='ra_2_2')         #256x35x35
        ra[1] = self.conv_factory(model, ra[1], 256, 384, kernel=3, stride=2, name='ra_2_3')      #384x17x17

        ra[2] = brew.max_pool(model, block35, 'ra_3_1_pool', kernel=3, stride=2)                  #320x17x17

        ra = brew.concat(model, ra, blob_out='ra')                                                #1088x35x35
        #
        block17 = ra
        for i in range(20):
            block17 = self.block17(model, block17, num_in_channels=1088,
                                   scale=0.1, name='inception_resnet_v2_b%d'%(i+1))               #1088x35x35
        # rb -reduction_b
        rb = [None, None, None, None]

        rb[0] = self.conv_factory(model, block17, 1088, 256, kernel=1, name='rb_1_1')             #256x17x17
        rb[0] = self.conv_factory(model, rb[0], 256, 384, kernel=3, stride=2, name='rb_1_2')      #384x8x8

        rb[1] = self.conv_factory(model, block17, 1088, 256, kernel=1, name='rb_2_1')             #256x17x17
        rb[1] = self.conv_factory(model, rb[1], 256, 288, kernel=3, stride=2, name='rb_2_2')      #288x8x8

        rb[2] = self.conv_factory(model, block17, 1088, 256, kernel=1, name='rb_3_1')             #256x17x17
        rb[2] = self.conv_factory(model, rb[2], 256, 288, kernel=3, pad=1, name='rb_3_2')         #288x17x17
        rb[2] = self.conv_factory(model, rb[2], 288, 320, kernel=3, stride=2, name='rb_3_3')      #320x8x8

        rb[3] = brew.max_pool(model, block17, 'rb_4_1_pool', kernel=3, stride=2)                  #1088x8x8

        rb = brew.concat(model, rb, blob_out='rb')                                                #2080x8x8
        #
        block8 = rb
        for i in range(9):
            block8 = self.block8(model, block8, num_in_channels=2080,
                                 scale=0.2, name='inception_resnet_v2_c%d'%(i+1))                 #2080x8x8
        block8 = self.block8(model, block8, num_in_channels=2080, relu=False,
                             name='inception_resnet_v2_c10')                                      #2080x8x8
        #
        conv6 = self.conv_factory(model, block8, 2080, 1536, kernel=1, name='conv6')              #1536x8x8
        pool8 = brew.average_pool(model, conv6, 'pool8', kernel=8, global_pool=True)              #1536x1x1
        drop8 = brew.dropout(model, pool8, 'dtop8', ratio=0.2,                                    #1536x1x1
                             is_test=(self.phase == 'inference'))

        if not self.__run_with_resnet50_trainer:
            return self.add_head_nodes(model, drop8, 1536, 'classifier', loss_scale=loss_scale)
        else:
            return brew.fc(model, drop8, 'classifier', dim_in=1536, dim_out=self.num_classes)


def create_inception_resnet_v2(model, data_node, num_input_channels=3, num_labels=1000,
                               no_bias=True, no_loss=True):
    """ This function should be used to create the Inception ResNet V2 model
    in resnet50_trainer.py. Find in that file the function 'create_resnet50_model_ops'
    and replace there call to 'resnet.create_resnet50' with this one 'create_inception_resnet_v2'.
    """
    if data_node != 'data':
        raise ValueError("Input data node name must be 'data'. This can be fixed.")
    if num_input_channels != 3:
        raise ValueError("Number of input channels must be 3. This can be fixed.")
    # The no_bias parameter is ignored.
    if no_loss != True:
        raise ValueError("The 'no_loss' parameter must be False. This can be fixed.")
    #
    # The 'batch_size' must be there but it will not be used since resnet50_trainer
    # has its own functions to feed data.
    params = {
        'input_shape': (num_input_channels, 299, 299),
        'num_classes': num_labels,
        'arg_scope': {'order': 'NCHW'},
        'batch_size': 16,
        'phase': 'training',
        'run_with_resnet50_trainer': True
    }
    model_builder = InceptionResNetV2(params)
    return model_builder.forward_pass_builder(model)
