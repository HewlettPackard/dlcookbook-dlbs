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
"""Unit tests for dlbs.builder.Builder class."""
import unittest
# append parent directory to import path
import env  #pylint: disable=W0611
# now we can import the lib module
from dlbs.builder import Builder
from dlbs.processor import Processor

class TestBuilder(unittest.TestCase):

    def setUp(self):
        pass

    def test_builder_1(self):
        """dlbs  ->  TestBuilder::test_builder_1                         [Test for plan builder #1.]"""
        plan = Builder.build(
            {'parameters': {'exp.backend': "tf_cnn_benchmark", 'exp.model': 'vgg16'}},
            {},
            {}
        )
        self.assertListEqual(
            plan,
            [{'exp.backend': "tf_cnn_benchmark", 'exp.model': 'vgg16'}]
        )

    def test_builder_2(self):
        """dlbs  ->  TestBuilder::test_builder_2                         [Test for plan builder #2.]"""
        plan = Builder.build(
            {'parameters': {'exp.backend': "tf_cnn_benchmark", 'exp.model': 'vgg16'}},
            {'exp.device_batch': 16},
            {}
        )
        self.assertListEqual(
            plan,
            [{'exp.backend': "tf_cnn_benchmark", 'exp.model': 'vgg16', 'exp.device_batch': 16}]
        )

    def test_builder_3(self):
        """dlbs  ->  TestBuilder::test_builder_3                         [Test for plan builder #3.]"""
        plan = Builder.build(
            {
                'parameters': {'exp.backend': "tf_cnn_benchmark", 'exp.model': 'vgg16'},
                'variables': {'exp.device_batch': [16, 32]}
            },
            {},
            {}
        )
        self.assertListEqual(
            plan,
            [
                {'exp.backend': "tf_cnn_benchmark", 'exp.model': 'vgg16', 'exp.device_batch': 16},
                {'exp.backend': "tf_cnn_benchmark", 'exp.model': 'vgg16', 'exp.device_batch': 32}
            ]
        )

    def test_builder_4(self):
        """dlbs  ->  TestBuilder::test_builder_4                         [Test for plan builder #4.]"""
        plan = Builder.build(
            {
                'parameters': {'exp.backend':'tf_cnn_benchmark', 'exp.model': 'vgg16'},
                'extensions': [
                    {
                        'condition':{'exp.backend': "tf_cnn_benchmark"},
                        'parameters': {'exp.device_batch': 128}
                    }
                ]
            },
            {},
            {}
        )
        Processor().compute_variables(plan)
        self.assertListEqual(
            plan,
            [
                {'exp.backend': "tf_cnn_benchmark", 'exp.model': 'vgg16', 'exp.device_batch': 128}
            ]
        )

    def test_builder_5(self):
        """dlbs  ->  TestBuilder::test_builder_5                         [Test for plan builder #5.]"""
        plan = Builder.build(
            {
                'parameters': {'exp.backend':'tf_cnn_benchmark', 'exp.device_batch': 256},
                'variables': {'exp.model': ['vgg16', 'text_cnn']},
                'extensions': [
                    {
                        'condition':{'exp.backend': "tf_cnn_benchmark", 'exp.model': 'text_cnn'},
                        'parameters': {'exp.device_batch': 512}
                    }
                ]
            },
            {},
            {}
        )
        Processor().compute_variables(plan)
        self.assertListEqual(
            plan,
            [
                {'exp.backend': "tf_cnn_benchmark", 'exp.model': 'vgg16', 'exp.device_batch': 256},
                {'exp.backend': "tf_cnn_benchmark", 'exp.model': 'text_cnn', 'exp.device_batch': 512}
            ]
        )

    def test_builder_6(self):
        """dlbs  ->  TestBuilder::test_builder_6                         [Test for plan builder #6.]"""
        sortedlist =lambda y: sorted(y , key=lambda elem: "{} {} {}".format(elem['exp.backend'], elem['exp.model'],elem['exp.replica_batch']))
        plan = Builder.build(
            {
                'parameters': {'exp.replica_batch': 256},
                'variables': {
                    'exp.backend':['tf_cnn_benchmark', 'caffe2'],
                    'exp.model': ['vgg16', 'text_cnn']
                },
                'extensions': [
                    {
                        'condition':{'exp.backend': "tf_cnn_benchmark", 'exp.model': 'text_cnn'},
                        'parameters': {'exp.replica_batch': 512}
                    }
                ]
            },
            {},
            {}
        )
        comparison = sortedlist([{'exp.backend': 'tf_cnn_benchmark', 'exp.model': 'vgg16', 'exp.replica_batch': 256},
             {'exp.backend': 'tf_cnn_benchmark', 'exp.model': 'text_cnn', 'exp.replica_batch': 512},
             {'exp.backend': 'caffe2', 'exp.model': 'vgg16', 'exp.replica_batch': 256},
             {'exp.backend': 'caffe2', 'exp.model': 'text_cnn', 'exp.replica_batch': 256}])
        Processor().compute_variables(plan)
        plan=sortedlist(plan)
        self.assertListEqual(plan,comparison)

    def test_builder_7(self):
        """dlbs  ->  TestBuilder::test_builder_7                         [Test for plan builder #7.]"""
        plan = Builder.build(
            {
                'parameters': {'exp.backend':'tf_cnn_benchmark', 'exp.model': 'vgg16'},
                'extensions': [
                    {
                        'condition':{'exp.backend': "tf_cnn_benchmark"},
                        'parameters': {'exp.device_batch': 128}
                    },
                    {
                        'condition':{'exp.backend': "tf_cnn_benchmark"},
                        'parameters': {'exp.disabled': 'true'}
                    }
                ]
            },
            {},
            {}
        )
        Processor().compute_variables(plan)
        self.assertListEqual(
            plan,
            [
                {'exp.backend': "tf_cnn_benchmark", 'exp.model': 'vgg16',
                 'exp.device_batch': 128, 'exp.disabled': 'true'}
            ]
        )

    def test_builder_8(self):
        """dlbs  ->  TestBuilder::test_builder_8                         [Test for plan builder #8.]"""
        plan = Builder.build(
            {
                'parameters': {'exp.backend':'bvlc_caffe', 'exp.model': 'vgg16'},
                'extensions': [
                    {
                        'condition':{'exp.backend': "([^_]+)_(.+)"},
                        'parameters': {
                            'exp.device_batch': 128,
                            'exp.backend_id': '${__condition.exp.backend_0}',   # bvlc_caffe
                            'exp.fork': '${__condition.exp.backend_1}',           # bvlc
                            'exp.backend': '${__condition.exp.backend_2}'       # caffe
                        }
                    }
                ]
            },
            {},
            {}
        )
        Processor().compute_variables(plan)
        self.assertListEqual(
            plan,
            [
                {'exp.backend': "caffe", 'exp.model': 'vgg16', 'exp.device_batch': 128,
                 'exp.backend_id': 'bvlc_caffe', 'exp.fork': 'bvlc'}
            ]
        )

    def test_builder_9(self):
        """dlbs  ->  TestBuilder::test_builder_9                         [Test for plan builder #9.]"""
        plan = Builder.build(
            {
                'parameters': {
                    'exp.backend':'bvlc_caffe',
                    'exp.model': 'vgg16',
                    'exp.path': '${${exp.fork}_caffe.path}',
                    'bvlc_caffe.path': '/opt/caffe'
                },
                'extensions': [
                    {
                        'condition':{'exp.backend': "([^_]+)_(.+)"},
                        'parameters': {
                            'exp.device_batch': 128,
                            'exp.backend_id': '${__condition.exp.backend_0}',   # bvlc_caffe
                            'exp.fork': '${__condition.exp.backend_1}',           # bvlc
                            'exp.backend': '${__condition.exp.backend_2}'       # caffe
                        }
                    }
                ]
            },
            {},
            {}
        )
        Processor().compute_variables(plan)
        self.assertListEqual(
            plan,
            [
                {'exp.backend': "caffe", 'exp.model': 'vgg16', 'exp.device_batch': 128,
                 'exp.backend_id': 'bvlc_caffe', 'exp.fork': 'bvlc',
                 'bvlc_caffe.path': '/opt/caffe', 'exp.path': '/opt/caffe'}
            ]
        )

if __name__ == '__main__':
    unittest.main()
