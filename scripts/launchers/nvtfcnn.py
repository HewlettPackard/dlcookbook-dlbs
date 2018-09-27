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
            runcommand=\
                r'{runtime_launcher}  {mpirun_cmd} {exp_singularity_launcher} exec {tensorflow_singularity_args} {runtime_python} '.format(\
                   runtime_launcher=co.vdict['runtime_launcher'], mpirun_cmd=co.mpirun_cmd,
                   exp_singularity_launcher=co.vdict['exp_singularity_launcher'],
                   tensorflow_singularity_args=co.vdict['tensorflow_singularity_args'],
                   runtime_python=co.vdict['runtime_python'])
        elif co.docker:
            runcommand=\
                r'{runtime_launcher} {exp_docker_launcher} run {tensorflow_docker_args} {mpirun_cmd} {runtime_python}'.format(\
                   runtime_launcher=co.vdict['runtime_launcher'],
                   exp_docker_launcher=co.vdict['exp_docker_launcher'],
                   tensorflow_docker_args=co.vdict['tensorflow_docker_args'],
                   mpirun_cmd=co.mpirun_cmd,
                   runtime_python=co.vdict['runtime_python'])
        else:
            runcommand=\
                r'{runtime_launcher} {mpirun_cmd} {runtime_python}'.format(\
                   runtime_launcher=co.vdict['runtime_launcher'],
                   mpirun_cmd=co.mpirun_cmd,
                   runtime_python=co.vdict['runtime_python'])
        script=\
                r'export {tensorflow_env}; '.format(tensorflow_env=co.vdict['tensorflow_env'])+ \
                r'echo -e "__results.start_time__= \x22$(date +%Y-%m-%d:%H:%M:%S:%3N)\x22"; '+ \
                r'{runcommand} {tensorflow_python_path}/benchmarks.py {nvtfcnn_args}; '.format(\
                       runcommand=runcommand,
                       tensorflow_python_path=co.vdict['tensorflow_python_path'],
                       nvtfcnn_args=co.vdict['nvtfcnn_args']) +\
                r'echo -e "__results.end_time__= \x22$(date +%Y-%m-%d:%H:%M:%S:%3N)\x22"'
    except Exception as e:
        print('Caught exception. Exiting.')
        traceback.print_exc()
        sys.exit(-1)

    co.run(script)

    ## Do some post-processing
    co.check_for_failed_run()
    co.logfile.close()

if __name__=='__main__':
    main()
