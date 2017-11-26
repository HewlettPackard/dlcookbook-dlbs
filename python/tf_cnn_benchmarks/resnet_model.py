# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Resnet model configuration.

References:
  Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
  Deep Residual Learning for Image Recognition
  arXiv:1512.03385 (2015)

  Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
  Identity Mappings in Deep Residual Networks
  arXiv:1603.05027 (2016)

  Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy,
  Alan L. Yuille
  DeepLab: Semantic Image Segmentation with Deep Convolutional Nets,
  Atrous Convolution, and Fully Connected CRFs
  arXiv:1606.00915 (2016)
"""

from six.moves import xrange  # pylint: disable=redefined-builtin

import model as model_lib


class ResNet(model_lib.Model):
    """ResNet models consistent with MXNet/Caffe2 implementation that follow
       recommended 'bn-relu-conv' order.
    """
    specs = {
        'resnet18':  { 'name': 'ResNet18',  'units': [2, 2, 2, 2],   'num_layers': 18  },  # pylint: disable=C0326
        'resnet34':  { 'name': 'ResNet34',  'units': [3, 4, 6, 3],   'num_layers': 34  },  # pylint: disable=C0326
        'resnet50':  { 'name': 'ResNet50',  'units': [3, 4, 6, 3],   'num_layers': 50  },  # pylint: disable=C0326
        'resnet101': { 'name': 'ResNet101', 'units': [3, 4, 23, 3],  'num_layers': 101 },  # pylint: disable=C0326
        'resnet152': { 'name': 'ResNet152', 'units': [3, 8, 36, 3],  'num_layers': 152 },  # pylint: disable=C0326
        'resnet200': { 'name': 'ResNet200', 'units': [3, 24, 36, 3], 'num_layers': 200 },  # pylint: disable=C0326
        'resnet269': { 'name': 'ResNet269', 'units': [3, 30, 48, 8], 'num_layers': 269 }   # pylint: disable=C0326
    }

    def __init__(self, model):
        super(ResNet, self).__init__(ResNet.specs[model]['name'], 224, 4, 0.005)
        self.model = model

    def add_inference(self, cnn):
        # Prepare config
        specs = ResNet.specs[self.model]
        if specs['num_layers'] >= 50:
            filter_list = [64, 256, 512, 1024, 2048]
            bottle_neck = True
        else:
            filter_list = [64, 64, 128, 256, 512]
            bottle_neck = False
        num_stages = 4
        units = specs['units']

        cnn.use_batch_norm = True
        cnn.batch_norm_config = {'decay': 0.9, 'epsilon': 2e-5, 'scale': True}
        # Conv will add batch norm and relu as well
        cnn.pad(p=3)
        cnn.conv(filter_list[0], 7, 7, 2, 2, mode='VALID')  # should be - pad = (3, 3)
        cnn.mpool(3, 3, 2, 2)

        for i in range(num_stages):
            cnn.residual_unit(num_filter=filter_list[i+1], stride=1 if i == 0 else 2,
                              dim_match=False, bottle_neck=bottle_neck)
            for _ in range(units[i] - 1):
                cnn.residual_unit(num_filter=filter_list[i+1], stride=1,
                                  dim_match=True, bottle_neck=bottle_neck)
        cnn.apool(7, 7, 1, 1)
        cnn.reshape([-1, filter_list[-1] * 1 * 1])


# Original implementation
class Resnetv1Model(model_lib.Model):
    """Resnet V1 cnn network configuration."""

    def __init__(self, model, layer_counts):
        defaults = {
            'resnet50': 64,
            'resnet101': 32,
            'resnet152': 32
        }
        batch_size = defaults.get(model, 32)
        super(Resnetv1Model, self).__init__(model, 224, batch_size, 0.005,
                                            layer_counts)

    def add_inference(self, cnn):
        """ For layers >= 50, bottlenck depth = depth / 4"""
        if self.layer_counts is None:
            raise ValueError('Layer counts not specified for %s' % self.get_model())

        cnn.use_batch_norm = True
        cnn.batch_norm_config = {'decay': 0.997, 'epsilon': 1e-5, 'scale': True}
        cnn.conv(64, 7, 7, 2, 2, mode='SAME_RESNET')
        cnn.mpool(3, 3, 2, 2)
        for _ in xrange(self.layer_counts[0]):
            cnn.resnet_bottleneck_v1(256, 64, 1)   # (depth, depth_bottleneck, stride)
        for i in xrange(self.layer_counts[1]):
            stride = 2 if i == 0 else 1
            cnn.resnet_bottleneck_v1(512, 128, stride)
        for i in xrange(self.layer_counts[2]):
            stride = 2 if i == 0 else 1
            cnn.resnet_bottleneck_v1(1024, 256, stride)
        for i in xrange(self.layer_counts[3]):
            stride = 2 if i == 0 else 1
            cnn.resnet_bottleneck_v1(2048, 512, stride)
        cnn.spatial_mean()
