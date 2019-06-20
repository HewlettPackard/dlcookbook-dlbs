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
"""Tests Helper class."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import unittest
# append parent directory to import path
import time
import dlbs.tests.env  # pylint: disable=W0611
from dlbs.utils import ResourceMonitor


class TestResourceMonitor(unittest.TestCase):
    def setUp(self):
        self.pid_folder = '/dev/shm/'

    def test_monitor(self):
        print ('Launchng monitor')
        monitor = ResourceMonitor(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "../../../scripts/resource_monitor.sh"
            ),
            self.pid_folder,
            0.1,
            "time:str:1,mem_virt:float:2,mem_res:float:3,mem_shrd:float:4,cpu:float:5,mem:float:6,power:float:7,gpus:float:8:"
        )
        monitor.run()
        for i in [1, 2]:
            print ('Initial cleaning PID file for workload %d' % i)
            monitor.empty_pid_file()
            print ('Launching workload %d' % i)
            time.sleep(2)
            print ('Sending workload (%d) pid' % i)
            monitor.write_pid_file(os.getpid())
            print ('Starting main workload %d' % i)
            time.sleep(5)
            print ('Final cleaning PID file for workload %d' % i)
            monitor.empty_pid_file()
            print ('Processing workload %d data' % i)
            timeseries = monitor.get_measurements()
            print(str(timeseries))
        print ('Stopping monitor')
        monitor.stop()

        print ('Monitor stopped')


if __name__ == '__main__':
    unittest.main()
