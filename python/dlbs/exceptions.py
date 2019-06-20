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
""" Module implements exception classes that can be thrown by the Deep Learning Benchmarking Suite."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class DLBSError(Exception):
    """Base class for all exceptions."""
    pass


class ConfigurationError(DLBSError):
    """This exception is thrown whenever error is found in an input configuration.

    Several examples of situations in which this exception gets thrown:

    - Cyclic dependency is found during variable expansion.
    - Variable cannot be expanded.
    - Un-parsable JSON value found in an input configuration.
    """
    pass


class LogicError(DLBSError):
    """This exception indicates a bug in a program.

    This exception in theory must never be thrown unless there is a bug in the
    program.
    """
    pass
