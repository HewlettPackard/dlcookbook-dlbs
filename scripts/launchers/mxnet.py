#!/usr/bin/env python
import sys
import traceback
import subprocess
from launcherutils import launcherutils

def main():
    try:
        co=launcherutils(sys.argv)
        print('__exp.framework_title__="MXNet"',file=co.logfile)

        if co.singularity:
            run_command=\
                r'{runtime_launcher}  {exp_singularity_launcher} exec {mxnet_singularity_args} '.format(\
                   runtime_launcher=co.vdict['runtime_launcher'],
                   exp_singularity_launcher=co.vdict['exp_singularity_launcher'],
                   mxnet_singularity_args=co.vdict['mxnet_singularity_args'])
        elif co.docker:
            run_command=\
                r'{runtime_launcher} {exp_docker_launcher} run {mxnet_docker_args}'.format(\
                   runtime_launcher=co.vdict['runtime_launcher'],
                   exp_docker_launcher=co.vdict['exp_docker_launcher'],
                   mxnet_docker_args=co.vdict['mxnet_docker_args'])
        else:
            run_command=\
                r'{runtime_launcher} {runtime_python}'.format(\
                   runtime_launcher=co.vdict['runtime_launcher'])

        benchmark_command = '{mxnet_bench_path}/benchmarks.py {mxnet_args}; '.format(\
                              mxnet_bench_path=co.vdict['mxnet_bench_path'],
                              mxnet_args=co.vdict['mxnet_args'])
    except Exception as e:
        print('Caught exception. Exiting.')
        traceback.print_exc()
        sys.exit(-1)

    co.run(run_command,benchmark_command)

if __name__=='__main__':
    main()
