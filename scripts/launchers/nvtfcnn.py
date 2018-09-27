#!/lvol/sfleisch/anaconda3/bin/python3
import sys
import traceback
import subprocess
from launcherutils import launcherutils

def main():
    try:
        co=launcherutils(sys.argv)
        co.setup_mpirun()
        print('__exp.framework_title__="TensorFlow"',file=co.logfile)

        if co.singularity:
            run_command=\
                r'{runtime_launcher}  {mpirun_cmd} {exp_singularity_launcher} exec {tensorflow_singularity_args} {runtime_python} '.format(\
                   runtime_launcher=co.vdict['runtime_launcher'], mpirun_cmd=co.mpirun_cmd,
                   exp_singularity_launcher=co.vdict['exp_singularity_launcher'],
                   tensorflow_singularity_args=co.vdict['tensorflow_singularity_args'],
                   runtime_python=co.vdict['runtime_python'])
        elif co.docker:
            run_command=\
                r'{runtime_launcher} {exp_docker_launcher} run {tensorflow_docker_args} {mpirun_cmd} {runtime_python}'.format(\
                   runtime_launcher=co.vdict['runtime_launcher'],
                   exp_docker_launcher=co.vdict['exp_docker_launcher'],
                   tensorflow_docker_args=co.vdict['tensorflow_docker_args'],
                   mpirun_cmd=co.mpirun_cmd,
                   runtime_python=co.vdict['runtime_python'])
        else:
            run_command=\
                r'{runtime_launcher} {mpirun_cmd} {runtime_python}'.format(\
                   runtime_launcher=co.vdict['runtime_launcher'],
                   mpirun_cmd=co.mpirun_cmd,
                   runtime_python=co.vdict['runtime_python'])

        benchmark_command = '{tensorflow_bench_path}/benchmarks.py {nvtfcnn_args}; '.format(\
                              tensorflow_bench_path=co.vdict['tensorflow_bench_path'],
                              nvtfcnn_args=co.vdict['nvtfcnn_args'])
    except Exception as e:
        print('Caught exception. Exiting.')
        traceback.print_exc()
        sys.exit(-1)

    co.run(co.vdict['tensorflow_env'],run_command,benchmark_command)

if __name__=='__main__':
    main()
