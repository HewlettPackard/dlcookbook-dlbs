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

"""Model configurations for CNN benchmarks.
"""

import alexnet_model
import googlenet_model
import inception_model
import lenet_model
import overfeat_model
import resnet_model
import trivial_model
import vgg_model
import deepmnist_model
import engacoustic_model
import sensornet_model

model_titles = {
    'deep_mnist': 'DeepMNIST',
    'eng_acoustic_model': 'EngAcousticModel',
    'sensor_net': 'SensorNet',
    'vgg11': 'VGG11',
    'vgg13': 'VGG13',
    'vgg16': 'VGG16',
    'vgg19': 'VGG19',
    'lenet': 'LeNet',
    'googlenet': 'GoogleNet',
    'overfeat': 'Overfeat',
    'alexnet': 'AlexNet',
    'trivial': 'Trivial',
    'inception3': 'InceptionV3',
    'inception4': 'InceptionV4',
    'resnet18': 'ResNet18',
    'resnet34': 'ResNet34',
    'resnet50': 'ResNet50',
    'resnet101': 'ResNet101',
    'resnet152': 'ResNet152',
    'resnet200': 'ResNet200',
    'resnet269': 'ResNet269'
}

def get_model_config(model):
  """Map model name to model network configuration."""
  if model == 'deep_mnist':
    mc = deepmnist_model.DeepMNISTModel()
  elif model == 'eng_acoustic_model':
    mc = engacoustic_model.EngAcousticModel()
  elif model == 'sensor_net':
    mc = sensornet_model.SensorNetModel()
  elif model == 'vgg11':
    mc = vgg_model.Vgg11Model()
  elif model == 'vgg13':
    mc = vgg_model.Vgg13Model()    
  elif model == 'vgg16':
    mc = vgg_model.Vgg16Model()
  elif model == 'vgg19':
    mc = vgg_model.Vgg19Model()
  elif model == 'lenet':
    mc = lenet_model.Lenet5Model()
  elif model == 'googlenet':
    mc = googlenet_model.GooglenetModel()
  elif model == 'overfeat':
    mc = overfeat_model.OverfeatModel()
  elif model == 'alexnet':
    mc = alexnet_model.AlexnetModel()
  elif model == 'trivial':
    mc = trivial_model.TrivialModel()
  elif model == 'inception3':
    mc = inception_model.Inceptionv3Model()
  elif model == 'inception4':
    mc = inception_model.Inceptionv4Model()
  elif model in ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnet200', 'resnet269'):
    mc = resnet_model.ResNet(model)
  else:
    raise KeyError('Invalid model name \'%s\'' % model)
  return mc
