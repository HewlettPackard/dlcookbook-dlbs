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
"""These unit tests test dlbs.utils.DictUtils class methods."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
# append parent directory to import path
import dlbs.tests.env   # pylint: disable=W0611
# now we can import the lib module
from dlbs.utils import DictUtils


class TestDictUtils(unittest.TestCase):

    def setUp(self):
        self.framework = 'TensorFlow'
        self.model = ["ResNet50", "ResNet101", "ResNet152"]
        self.device_batch = 16
        self.dictionary = {
            'exp.framework': self.framework,
            'exp.model': self.model,
            'exp.device_batch': self.device_batch
        }

    def test_ensure_exists_1(self):
        """dlbs  ->  TestDictUtils::test_ensure_exists_1                 [Testing dictionary helpers #1]"""
        DictUtils.ensure_exists(self.dictionary, 'exp.framework')

        self.assertEqual('exp.framework' in self.dictionary, True)
        self.assertEqual('exp.model' in self.dictionary, True)
        self.assertEqual('exp.device_batch' in self.dictionary, True)

        self.assertEqual(self.dictionary['exp.framework'], self.framework)
        self.assertEqual(self.dictionary['exp.model'], self.model)
        self.assertEqual(self.dictionary['exp.device_batch'], self.device_batch)

        self.assertEqual(len(self.dictionary), 3)

    def test_ensure_exists_2(self):
        """dlbs  ->  TestDictUtils::test_ensure_exists_2                 [Testing dictionary helpers #2]"""
        DictUtils.ensure_exists(self.dictionary, 'exp.effective_batch')

        self.assertEqual('exp.framework' in self.dictionary, True)
        self.assertEqual('exp.model' in self.dictionary, True)
        self.assertEqual('exp.device_batch' in self.dictionary, True)
        self.assertEqual('exp.effective_batch' in self.dictionary, True)

        self.assertEqual(self.dictionary['exp.framework'], self.framework)
        self.assertEqual(self.dictionary['exp.model'], self.model)
        self.assertEqual(self.dictionary['exp.device_batch'], self.device_batch)
        self.assertEqual(self.dictionary['exp.effective_batch'], None)

        self.assertEqual(len(self.dictionary), 4)

    def test_ensure_exists_3(self):
        """dlbs  ->  TestDictUtils::test_ensure_exists_3                 [Testing dictionary helpers #3]"""
        DictUtils.ensure_exists(self.dictionary, 'exp.data_dir', '/nfs/imagenet')

        self.assertEqual('exp.framework' in self.dictionary, True)
        self.assertEqual('exp.model' in self.dictionary, True)
        self.assertEqual('exp.device_batch' in self.dictionary, True)
        self.assertEqual('exp.data_dir' in self.dictionary, True)

        self.assertEqual(self.dictionary['exp.framework'], self.framework)
        self.assertEqual(self.dictionary['exp.model'], self.model)
        self.assertEqual(self.dictionary['exp.device_batch'], self.device_batch)
        self.assertEqual(self.dictionary['exp.data_dir'], '/nfs/imagenet')

        self.assertEqual(len(self.dictionary), 4)

    def test_lists_to_strings_1(self):
        """dlbs  ->  TestDictUtils::test_lists_to_strings_1              [Testing lists-to-strings helpers #1]"""
        DictUtils.lists_to_strings(self.dictionary)

        self.assertEqual('exp.framework' in self.dictionary, True)
        self.assertEqual('exp.model' in self.dictionary, True)
        self.assertEqual('exp.device_batch' in self.dictionary, True)

        self.assertEqual(self.dictionary['exp.framework'], self.framework)
        self.assertEqual(self.dictionary['exp.model'], "ResNet50 ResNet101 ResNet152")
        self.assertEqual(self.dictionary['exp.device_batch'], self.device_batch)

        self.assertEqual(len(self.dictionary), 3)

    def test_lists_to_strings_2(self):
        """dlbs  ->  TestDictUtils::test_lists_to_strings_2              [Testing lists-to-strings helpers #2]"""
        DictUtils.lists_to_strings(self.dictionary, separator=';')

        self.assertEqual('exp.framework' in self.dictionary, True)
        self.assertEqual('exp.model' in self.dictionary, True)
        self.assertEqual('exp.device_batch' in self.dictionary, True)

        self.assertEqual(self.dictionary['exp.framework'], self.framework)
        self.assertEqual(self.dictionary['exp.model'], "ResNet50;ResNet101;ResNet152")
        self.assertEqual(self.dictionary['exp.device_batch'], self.device_batch)

        self.assertEqual(len(self.dictionary), 3)

    def test_match_1(self):
        """dlbs  ->  TestDictUtils::test_match_1                         [Testing matching helpers #1]"""
        for frameworks in [self.framework, [self.framework], [self.framework, "Caffe2"]]:
            # We can match against existing key with strict policy
            self.assertEquals(
                DictUtils.match(self.dictionary, {'exp.framework': frameworks}, policy='strict'),
                True
            )
            # We cannot match against non existing key with strict policy
            self.assertEquals(
                DictUtils.match(
                    self.dictionary, {'exp.framework_id': self.framework}, policy='strict'
                ),
                False
            )
            # We can match against non existing key with relaxed policy
            self.assertEquals(
                DictUtils.match(
                    self.dictionary, {'exp.framework_id': self.framework}, policy='relaxed'
                ),
                True
            )
        # Key exist, different values
        self.assertEquals(
            DictUtils.match(self.dictionary, {'exp.framework': 'Caffe2'}, policy='strict'),
            False
        )
        # AND condition + strict policy
        self.assertEquals(
            DictUtils.match(self.dictionary, {'exp.framework': self.framework, 'exp.device_batch': self.device_batch}, policy='strict'),
            True
        )
        # AND condition
        self.assertEquals(
            DictUtils.match(self.dictionary, {'exp.framework': [self.framework, 'Caffe2'], 'exp.device_batch': self.device_batch}, policy='strict'),
            True
        )
        self.assertEquals(
            DictUtils.match(self.dictionary, {'exp.framework': self.framework, 'exp.device_batch': 2*self.device_batch}, policy='strict'),
            False
        )
        # AND condition relaxed policy
        self.assertEquals(
            DictUtils.match(self.dictionary, {'exp.framework': self.framework, 'exp.effective_batch': 2*self.device_batch}, policy='relaxed'),
            True
        )
        # AND condition
        self.assertEquals(
            DictUtils.match(self.dictionary, {'exp.framework': [self.framework, 'Caffe2'], 'exp.effective_batch': 2*self.device_batch}, policy='relaxed'),
            True
        )
        # Relaxed policy with multiple fields that exist and do not match
        self.assertEquals(
            DictUtils.match(self.dictionary, {'exp.framework': self.framework, 'exp.device_batch': 2*self.device_batch}, policy='relaxed'),
            False
        )

    def test_match_2(self):
        """dlbs  ->  TestDictUtils::test_match_2                         [Testing matching helpers #2]"""
        dictionary = {'exp.framework': "bvlc_caffe", 'exp.model': "ResNet150"}
        matches = {}
        self.assertEquals(
            DictUtils.match(dictionary, {'exp.model': r'([^\d]+)(\d+)'}, policy='strict', matches=matches),
            True
        )
        self.assertEquals(len(matches), 3)
        self.assertEquals(matches['exp.model_0'], 'ResNet150')
        self.assertEquals(matches['exp.model_1'], 'ResNet')
        self.assertEquals(matches['exp.model_2'], '150')

    def test_match_3(self):
        """dlbs  ->  TestDictUtils::test_match_3                         [Testing matching helpers #3]"""
        dictionary = {'exp.framework': "bvlc_caffe", 'exp.model': "ResNet150"}
        matches = {}
        self.assertEquals(
            DictUtils.match(dictionary, {'exp.framework': '([^_]+)_(.+)'}, policy='strict', matches=matches),
            True
        )
        self.assertEquals(len(matches), 3)
        self.assertEquals(matches['exp.framework_0'], 'bvlc_caffe')
        self.assertEquals(matches['exp.framework_1'], 'bvlc')
        self.assertEquals(matches['exp.framework_2'], 'caffe')

    def test_match_4(self):
        """dlbs  ->  TestDictUtils::test_match_4                         [Testing matching helpers #4]"""
        dictionary = {'exp.framework': "bvlc_caffe", 'exp.model': "ResNet150"}
        self.assertEquals(
            DictUtils.match(dictionary, {'exp.framework': '([^_]+)_(.+)'}, policy='strict'),
            True
        )

    def test_match_5(self):
        """dlbs  ->  TestDictUtils::test_match_5                         [Testing matching helpers #5]"""
        dictionary = {'exp.framework': "bvlc_caffe", 'exp.model': "ResNet150"}
        matches = {}
        self.assertEquals(
            DictUtils.match(dictionary, {'exp.framework': '([^_]+)D(.+)'}, policy='strict', matches=matches),
            False
        )
        self.assertEquals(len(matches), 0)

    def test_match_6(self):
        """Test empty strings can match"""
        dictionary = {'exp.framework': "bvlc_caffe", 'exp.data_dir': ""}
        #
        matches = {}
        for val in ('', ' ', '  ', '    '):
            self.assertEquals(
                DictUtils.match(dictionary, {'exp.framework': val}, policy='strict', matches=matches),
                False
            )
            self.assertEqual(len(matches), 0)
        #
        self.assertEquals(
            DictUtils.match(dictionary, {'exp.data_dir': ''}, policy='strict', matches=matches),
            True
        )
        self.assertEqual(len(matches), 1)
        self.assertIn('exp.data_dir_0', matches)
        self.assertEqual(matches['exp.data_dir_0'], '')


if __name__ == '__main__':
    unittest.main()
