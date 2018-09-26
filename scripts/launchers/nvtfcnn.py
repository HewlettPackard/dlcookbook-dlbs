#!/lvol/sfleisch/anaconda3/bin/python3
import sys
import traceback
import subprocess
import launchers

def main():
    co=launchers.commonLauncherClass(sys.argv)
    
    print('__exp.framework_title__="TensorFlow"',file=co.logfile)

   
    if co.check_key('exp_mpirun_hosts'): co.vdict['exp_mpirun_args']="-H {} ".format(co.vdict['exp_mpirun_hosts']+co.vdict['exp_mpirun_args'])
    if not co.check_key('exp_mpirun_num_tasks'): co.vdict['exp_mpirun_num_tasks']=1
    co.vdict['exp_mpirun_args']="-np {} ".format(co.vdict['exp_mpirun_num_tasks'])+co.vdict['exp_mpirun_args']
    try:
        co.assert_not_docker_and_singularity(co.logfatal)
        if co.test_for_true('exp_singularity',co.logfatal):
            co.assert_singularity_image_exists()
            runcommand=\
                r'{runtime_launcher}  {exp_mpirun} {exp_mpirun_args} {exp_singularity_launcher} exec {tensorflow_singularity_args} {runtime_python} '.format(\
                   runtime_launcher=co.vdict['runtime_launcher'],
                   exp_mpirun=co.vdict['exp_mpirun'],
                   exp_mpirun_args=co.vdict['exp_mpirun_args'],
                   exp_singularity_launcher=co.vdict['exp_singularity_launcher'],
                   tensorflow_singularity_args=co.vdict['tensorflow_singularity_args'],
                   runtime_python=co.vdict['runtime_python'])
        elif test_for_true('exp_docker',co.logfatal,co.vdict):
            assert_docker_image_exists(co.vdict)
            runcommand=\
                r'{runtime_launcher} {exp_docker_launcher} run {tensorflow_docker_args} {exp_mpirun} {exp_mpirun_args} {runtime_python}'.format(\
                   runtime_launcher=co.vdict['runtime_launcher'],
                   exp_docker_launcher=co.vdict['exp_docker_launcher'],
                   tensorflow_docker_args=co.vdict['tensorflow_docker_args'],
                   exp_mpirun=co.vdict['exp_mpirun'],
                   exp_mpirun_args=co.vdict['exp_mpirun_args'],
                   runtime_python=co.vdict['runtime_python'])
        else:
            runcommand=\
                r'{runtime_launcher} {exp_mpirun} {exp_mpirun_args} {runtime_python}'.format(\
                   runtime_launcher=co.vdict['runtime_launcher'],
                   exp_mpirun=co.vdict['exp_mpirun'],
                   exp_mpirun_args=co.vdict['exp_mpirun_args'],
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
        co.logfatal('Caught exception. Exiting.')
        co.logfile.close()
        traceback.print_exc()
        sys.exit(-1)
    co.run(script)
    #
    ## Do some post-processing
    #if tf_error ${exp_log_file}; then
    #    logwarn "error in \"${exp_log_file}\" with effective batch ${exp_effective_batch} (replica batch ${exp_replica_batch})";
    #    update_error_file "${__batch_file__}" "${exp_replica_batch}";
    #    echo "__exp.status__=\"failure\"" >> ${exp_log_file}
    #fi
if __name__=='__main__':
    main()

