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
"""Unit tests for testing caffe2_benchmarks.model_builder.ModelFactory class."""
import unittest
# append parent directory to import path
#import env #pylint: disable=W0611
# now we can import the lib module
from mxnet_benchmarks.model_factory import ModelFactory

def net_specs(name, input_shape, num_classes):
    """Returns object with net specs."""
    return {
        'name': name,
        'input_shape': input_shape,
        'num_classes': num_classes
    }

class TestModelFactory(unittest.TestCase):
    """Tests caffe2_benchmarks.model_builder.ModelBuilder class

    It checks for correst assignment of these class members: `batch_size`, `phase`,
    `name`, `input_shape` and `num_classes`.
    """

    def setUp(self):
        self.models = {}
        self.models['deep_mnist'] = net_specs('DeepMNIST', (784,), 10)
        self.models['eng_acoustic_model'] = net_specs('EngAcousticModel', (540,), 8192)
        self.models['sensor_net'] = net_specs('SensorNet', (784,), 16)
        self.models['alexnet'] = net_specs('AlexNet', (3, 227, 227), 1000)
        self.models['googlenet'] = net_specs('GoogleNet', (3, 224, 224), 1000)
        for layers in [11, 13, 16, 19]:
            self.models['vgg%d' % layers] = net_specs('VGG%d' % layers, (3, 224, 224), 1000)
        for layers in [18, 34, 50, 101, 152, 200, 269]:
            self.models['resnet%d' % layers] = net_specs('ResNet%d' % layers, (3, 224, 224), 1000)

    def test_import_models(self):
        """mxnet_benchmarks  ->  TestModelFactory::test_import_models    [Import neural network models.]"""
        model_ids = [
            'deep_mnist', 'eng_acoustic_model', 'sensor_net', 'alexnet', 'googlenet', 
            'vgg11', 'vgg13', 'vgg16', 'vgg19',
            'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnet200', 'resnet269'
        ]
        for model_id in model_ids:
            self.assertIn(model_id, ModelFactory.models)

    def test_model_builder(self):
        """mxnet_benchmarks  ->  TestModelFactory::test_model_builder    [Create neural network models.]"""
        for phase in ['inference', 'training']:
            for model in self.models:
                model_specs = self.models[model]
                model_builder = ModelFactory.get_model(
                    {'model': model, 'phase': phase}
                )
                self.assertIsNotNone(model_builder)
                self.assertEqual(model_builder.phase, phase)
                self.assertEqual(model_builder.name, model_specs['name'])
                self.assertEqual(model_builder.input_shape, model_specs['input_shape'])
                self.assertEqual(model_builder.num_classes, model_specs['num_classes'])

if __name__ == '__main__':
    unittest.main()
