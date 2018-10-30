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
""" Mxnet cluster launcher that is responsible for launching process on one node.

Mxnet cluster consists of three agents - one scheduler and multiple servers and
workers. This launcher runs a cluster with a specific topology:
1. There is only one scheduler running on some node.
2. Each node runs one worker and one server agents.
3. Each agent may use multiple GPUs.
4. All agents use the same number of GPUs (heterogeneous architecture).

Idea is similar to this one:
  https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py

The very fist agent that needs to run is the scheduler. Then all others can run.
"""
from __future__ import absolute_import
from __future__ import print_function
import argparse
import sys
import os
import subprocess
import time


def parse_args():
    """Parse command line arguments."""
    def str2bool(val):
        """Return true if val represents true  value."""
        return val.lower() in ('true', 'on', 't', '1')
    parser = argparse.ArgumentParser()
    parser.add_argument('--rendezvous', type=str, default='')
    parser.add_argument('--num_workers', type=str, default='1')
    parser.add_argument('--scheduler', nargs='?', const=True, default=False, type=str2bool)
    parser.add_argument('--verbose', type=str, default='0')
    parser.add_argument("benchmark_script", type=str)
    parser.add_argument('benchmark_args', nargs=argparse.REMAINDER)
    return parser.parse_args()


def print_now(msg):
    """Print message and flush standard output."""
    print(msg)
    sys.stdout.flush()


class Cluster(object):
    """MXNET Cluster manager."""
    def __init__(self, args=None):
        if args is not None:
            self.num_workers = args.num_workers
            self.num_servers = args.num_workers
            self.is_scheduler = args.scheduler
            self.verbose_level = args.verbose
            self.benchmark = {'script': args.benchmark_script, 'args': args.benchmark_args}

            scheduler = args.rendezvous.split(':')
            if len(scheduler) == 2:
                self.scheduler = {'interface': None, 'host': scheduler[0], 'port': scheduler[1]}
            elif len(scheduler) == 3:
                self.scheduler = {'interface': scheduler[0], 'host': scheduler[1], 'port': scheduler[2]}
            else:
                raise ValueError("Invalid rendezvous specifier format (%s)" % (args.rendezvous))
        self.agents = {
            'scheduler': None, 'server': None, 'worker': None
        }

    def agent_specs(self, role):
        """Return agent specs for a particular role."""
        specs = {
            'DMLC_ROLE': role, 'DMLC_PS_ROOT_URI': self.scheduler['host'],
            'DMLC_PS_ROOT_PORT': self.scheduler['port'], 'DMLC_NUM_SERVER': self.num_workers,
            'DMLC_NUM_WORKER': self.num_servers, 'PS_VERBOSE': self.verbose_level
        }
        if self.scheduler['interface'] is not None:
            specs['DMLC_INTERFACE'] = self.scheduler['interface']
        return specs

    def run_agent(self, role):
        """Run agent in a background process.
        Args:
            role: `str` An agent's role. One of 'scheduler' or 'server'.
        Returns:
            An instance of a subprocess.
        """
        env = os.environ.copy()
        cluster_vars = self.agent_specs(role)
        print_now("Running agent '%s' with cluster parameters '%s'" % (role, cluster_vars))
        env.update(cluster_vars)
        return subprocess.Popen(
            [sys.executable, '-u', '-c', 'import mxnet;'],
            shell=False,
            env=env
        )

    def launch(self):
        """Launch cluster."""
        if self.is_scheduler:
            self.agents['scheduler'] = self.run_agent('scheduler')
            time.sleep(5)
            if self.agents['scheduler'].poll() is not None:
                print_now(
                    "Scheduler was not started (return code=%s)" % (self.agents['scheduler'].poll())
                )
                exit(1)
        self.agents['server'] = self.run_agent('server')

        env = os.environ.copy()
        cluster_vars = self.agent_specs('worker')
        print_now("Running agent 'worker' with cluster parameters '%s'" % (cluster_vars))
        env.update(cluster_vars)
        cmd = [sys.executable, '-u', self.benchmark['script']] + self.benchmark['args']
        self.agents['worker'] = subprocess.Popen(cmd, env=env)

    def wait(self):
        """Wait for a worker to complete."""
        print_now("Waiting for a worker.")
        self.agents['worker'].wait()

    def shutdown(self):
        """Shutdown.

        Generally, scheduler and server should shutdown themselves once all workers exited.
        """
        def _alive(proc):
            return proc is not None and proc.poll() is None
        print_now("Shutting down agents.")
        if _alive(self.agents['server']):
            self.agents['server'].terminate()
        if _alive(self.agents['scheduler']):
            self.agents['scheduler'].terminate()


def main():
    """Runs agents on a local node."""
    cluster = Cluster(args=parse_args())
    cluster.launch()
    cluster.wait()
    cluster.shutdown()


if __name__ == "__main__":
    print_now("Running mxnet benchmarks with cluster launcher.")
    main()
