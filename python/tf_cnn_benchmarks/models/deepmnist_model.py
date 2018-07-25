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

"""DeepMNIST model configuration.

References:
  Ciresan et al. Neural Computation 10, 2010 and arXiv 1003.0358, 2010
  http://arxiv.org/abs/1003.0358
  We have 10 classes here (search tf_cnn_benchmarks.py for 'nclass')
  We also have one channel only (search tf_cnn_benchmarks.py for 'input_nchan =')
"""

from models import model


class DeepMNISTModel(model.Model):
  """DeepMNIST fully connected model."""

  def __init__(self):
    super(DeepMNISTModel, self).__init__('deep_mnist', 28, 512, 0.005)

  def add_inference(self, cnn):
    # We have one channel image of size 28x28
    cnn.reshape([-1, 28*28])
    cnn.affine(2500)
    cnn.affine(2000)
    cnn.affine(1500)
    cnn.affine(1500)
    cnn.affine(1000)
    cnn.affine(500)
