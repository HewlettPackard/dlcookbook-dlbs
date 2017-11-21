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
"""Unit tests for testing caffe2_benchmarks.data_iterator.DataIterator class."""
import unittest
from mxnet_benchmarks.data_iterator import SyntheticDataIterator
import numpy as np

class TestDataIterator(unittest.TestCase):

    def setUp(self):
        self.data_iter = SyntheticDataIterator(1000, (1024, 3, 227, 227), max_iter=500, dtype=np.float32)
        pass

    def test_sdi_constructor(self):
        """mxnet_benchmarks  ->  TestDataIterator::test_sdi_constructor  [SyntheticDataIterator constructor.]
        """
        self.assertEqual(self.data_iter.batch_size, 1024)
        self.assertEqual(self.data_iter.cur_iter, 0)
        self.assertEqual(self.data_iter.max_iter, 500)
        self.assertEqual(self.data_iter.dtype, np.float32)
        self.assertIsNotNone(self.data_iter.data)
        self.assertEqual(len(self.data_iter.data.shape), 4)
        self.assertIsNotNone(self.data_iter.label)
        self.assertEqual(len(self.data_iter.label.shape), 1)

    def test_sdi_iteration(self):
        """mxnet_benchmarks  ->  TestDataIterator::test_sdi_iteration    [SyntheticDataIterator iteration.]
        """
        num_iterations = 0
        for batch in self.data_iter:
            num_iterations += 1
        self.assertEqual(num_iterations, 500)

if __name__ == '__main__':
    unittest.main()
