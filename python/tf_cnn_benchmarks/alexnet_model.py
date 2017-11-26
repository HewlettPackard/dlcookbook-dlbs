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

"""Alexnet model configuration.

References:
  Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton
  ImageNet Classification with Deep Convolutional Neural Networks
  Advances in Neural Information Processing Systems. 2012
  
  http://ethereon.github.io/netscope/#/gist/5c94a074f4e4ac4b81ee28a796e04b5d
  Reference AlexNet with grouped convolutions removed.
"""

import model


class AlexnetModel(model.Model):
  """Alexnet cnn model."""

  def __init__(self):
    super(AlexnetModel, self).__init__('alexnet', 224 + 3, 512, 0.005)

  def add_inference(self, cnn):
    # def conv(self, num_out_channels, k_height, k_width, d_height=1, d_width=1, mode='SAME', ...)
    # Note: VALID requires padding the images by 3 in width and height
    cnn.conv(96, 11, 11, 4, 4, 'VALID')   # Originally 64 output channels
    cnn.lrn(5, 0.0001, 0.75)              # Was not here originally
    cnn.mpool(3, 3, 2, 2)
    cnn.conv(256, 5, 5)                   # Originally, 192 output channels.
    cnn.lrn(5, 0.0001, 0.75)              # Was not here originally
    cnn.mpool(3, 3, 2, 2)
    cnn.conv(384, 3, 3)
    cnn.conv(384, 3, 3)
    cnn.conv(256, 3, 3)
    cnn.mpool(3, 3, 2, 2)
    cnn.reshape([-1, 256 * 6 * 6])
    cnn.affine(4096)
    cnn.dropout()
    cnn.affine(4096)
    cnn.dropout()
