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

"""EngAcousticModel model configuration.

References:
  A typical deep fully connected neural network used in hybrid HMM/DNN speech
  recognition system to model acustics (transition from MFCC coefficients to,
  usually, clustered tri-phones).
  We have 8192 classes here (search tf_cnn_benchmarks.py for 'nclass')
  Since we do not work with images, we model input signal by using images of
  size 1x23x23 hat gives 529 features. In other experiments, we work with 540
  features.
"""

from models import model


class AcousticModel(model.Model):
  """Acoustic Model (fully connected neural net)."""

  def __init__(self):
    super(AcousticModel, self).__init__('acoustic_model', 23, 512, 0.005)

  def add_inference(self, cnn):
    # We have one channel image of size 28x28
    cnn.reshape([-1, 23*23])
    cnn.affine(2048)
    cnn.affine(2048)
    cnn.affine(2048)
    cnn.affine(2048)
    cnn.affine(2048)

