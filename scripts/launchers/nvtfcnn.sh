#!/bin/bash
echo $1
unknown_params_action=set
. $DLBS_ROOT/scripts/parse_options.sh || exit 1;    # Parse command line options
exit
. $DLBS_ROOT/scripts/utils.sh
loginfo "$0 $*" >> ${exp_log_file}                  # Log command line arguments for debugging purposes
echo "__exp.framework_title__=\"TensorFlow\"" >> ${exp_log_file}
# Check batch is small enough for this experiment
__batch_file__="$(dirname ${exp_log_file})/${exp_framework}_${exp_device_type}_${exp_model}.batch"
# Thus, the environment must be initialized inside this scrip lazily.
[ -z "${runtime_launcher}" ] && runtime_launcher=":;"
assert_not_docker_and_singularity
if [ ! -z ${exp_mpirun_hosts} ]; then
	exp_mpirun_args="-H ${exp_mpirun_hosts} ${exp_mpirun_args}"
fi
if [ ! -z ${exp_mpirun_num_tasks} ]; then
    exp_mpirun_args="-np ${exp_mpirun_num_tasks} ${exp_mpirun_args}"
fi

if [ "${exp_docker}" = "true" ]; then
    assert_docker_img_exists ${exp_docker_image}
    export run_container="${exp_docker_launcher} run ${tensorflow_docker_args}"
    # Won't work for multi-host. Will require some work.
    # https://github.com/NVIDIA/nvidia-docker/wiki/Frequently-Asked-Questions#do-you-support-docker-compose:
    script="\
        export ${tensorflow_env};\
        echo -e \"__results.start_time__= \x22\$(date +%Y-%m-%d:%H:%M:%S:%3N)\x22\";\
        ${runtime_launcher}  ${run_container} ${exp_mpirun} ${exp_mpirun_args} ${runtime_python} ${tensorflow_python_path}/benchmarks.py ${nvtfcnn_args};\
        echo -e \"__results.end_time__= \x22\$(date +%Y-%m-%d:%H:%M:%S:%3N)\x22\";
"
elif [ "${exp_singularity}" = "true" ]; then
    assert_singularity_img_exists ${exp_singularity_image}
    export run_container="${exp_singularity_launcher} exec ${tensorflow_singularity_args}"
    script="\
        export ${tensorflow_env};\
        echo -e \"__results.start_time__= \x22\$(date +%Y-%m-%d:%H:%M:%S:%3N)\x22\";\
        ${runtime_launcher}  ${exp_mpirun} ${exp_mpirun_args} ${run_container} ${runtime_python} ${tensorflow_python_path}/benchmarks.py ${nvtfcnn_args};\
        echo -e \"__results.end_time__= \x22\$(date +%Y-%m-%d:%H:%M:%S:%3N)\x22\";
    "
else
    script="\
        export ${tensorflow_env};\
        echo -e \"__results.start_time__= \x22\$(date +%Y-%m-%d:%H:%M:%S:%3N)\x22\";\
        ${runtime_launcher}  ${exp_mpirun} ${exp_mpirun_args} ${runtime_python} ${tensorflow_python_path}/benchmarks.py ${nvtfcnn_args};\
        echo -e \"__results.end_time__= \x22\$(date +%Y-%m-%d:%H:%M:%S:%3N)\x22\";
    "
fi
echo script $script
/bin/bash -c "eval $script" >> ${exp_log_file} 2>&1

# Do some post-processing
if tf_error ${exp_log_file}; then
    logwarn "error in \"${exp_log_file}\" with effective batch ${exp_effective_batch} (replica batch ${exp_replica_batch})";
    update_error_file "${__batch_file__}" "${exp_replica_batch}";
    echo "__exp.status__=\"failure\"" >> ${exp_log_file}
fi
