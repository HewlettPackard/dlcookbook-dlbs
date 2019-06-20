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
"""Modifies sys path so that we can easily run unit tests."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os

# Path to this file is ... python/dlbs/tests/env.py
# https://stackoverflow.com/questions/61151/where-do-the-python-unit-tests-go
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../.."
    )
)
