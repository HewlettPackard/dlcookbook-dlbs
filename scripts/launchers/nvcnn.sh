#!/bin/bash
unknown_params_action=set
. $DLBS_ROOT/scripts/parse_options.sh || exit 1;    # Parse command line options
. $DLBS_ROOT/scripts/utils.sh
loginfo "$0 $*" >> ${exp_log_file}                  # Log command line arguments for debugging purposes
# Now, we have support only for training
if [ "${exp_phase}" == "inference" ]; then
  report_and_exit "failure" "NVCNN TensorFlow can only benchmark 'training' phase." "${exp_log_file}";
fi
echo "__exp.framework_title__=\"TensorFlow-nvcnn\"" >> ${exp_log_file}
if [ "$exp_status" = "simulate" ]; then
    echo "${nvcnn_env} ${runtime_launcher} python ${nvcnn_python_path}/nvcnn.py ${nvcnn_args}"
    exit 0
fi
# Check batch is small enough for this experiment
__batch_file__="$(dirname ${exp_log_file})/${exp_framework}_nvcnn_${exp_device_type}_${exp_model}.batch"
is_batch_good "${__batch_file__}" "${exp_replica_batch}" || {
  report_and_exit "skipped" "The replica batch size (${exp_replica_batch}) is too large for given SW/HW configuration." "${exp_log_file}";
}
# This script is to be executed inside docker container or on a host machine.
# Thus, the environment must be initialized inside this scrip lazily.
[ -z "${runtime_launcher}" ] && runtime_launcher=":;"
script="\
    export ${nvcnn_env};\
    echo -e \"__exp.framework_ver__= \x22\$(python -c 'import tensorflow as tf; print (tf.__version__);')\x22\";\
    echo -e \"__results.start_time__= \x22\$(date +%Y-%m-%d:%H:%M:%S:%3N)\x22\";\
    ${runtime_launcher} ${runtime_python} ${nvcnn_python_path}/nvcnn.py ${nvcnn_args} &\
    proc_pid=\$!;\
    [ \"${monitor_frequency}\" != \"0\" ] && echo -e \"\${proc_pid}\" > ${monitor_backend_pid_folder}/proc.pid;\
    wait \${proc_pid};\
    echo -e \"__results.end_time__= \x22\$(date +%Y-%m-%d:%H:%M:%S:%3N)\x22\";\
    echo -e \"__results.proc_pid__= \${proc_pid}\";\
"
if [ "${exp_docker}" = "true" ]; then
    assert_docker_img_exists ${exp_docker_image}
    ${exp_docker_launcher} run ${nvcnn_docker_args} /bin/bash -c "eval $script" >> ${exp_log_file} 2>&1
else
    eval $script >> ${exp_log_file} 2>&1
fi

# Do some post-processing
if tf_error ${exp_log_file}; then
    logwarn "error in \"${exp_log_file}\" with effective batch ${exp_effective_batch} (replica batch ${exp_replica_batch})";
    update_error_file "${__batch_file__}" "${exp_replica_batch}";
    echo "__exp.status__=\"failure\"" >> ${exp_log_file}
else
    :;
    # If everything's OK, we now can get IPS (throughput):
    # What needs be done: check for error
    #ips=$(grep "Images/sec:" ${exp_log_file}  | awk '{print $2}')
    #is_positive=$(echo "print($ips > 0)" | python)
    #if [ "$is_positive" == "True" ]; then
    #  tm=$(echo "print (${exp_effective_batch} * 1000.0 / $ips)" | python)
    #  echo -e "__results.time__= $tm" >> ${exp_log_file}
    #  echo -e "__results.throughput__= $ips" >> ${exp_log_file}
    #fi
fi
