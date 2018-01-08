#!/bin/bash
unknown_params_action=set
. $DLBS_ROOT/scripts/parse_options.sh || exit 1;    # Parse command line options
. $DLBS_ROOT/scripts/utils.sh
loginfo "$0 $*"  >> ${exp_log_file}                 # Log command line arguments for debugging purposes
if [ "$exp_simulation" = "true" ]; then
    echo "${runtime_bind_proc} python ${tensorflow_python_path}/tf_cnn_benchmarks.py ${tensorflow_args}"
    exit 0
fi
__framework__="TensorFlow"
__batch_file__="$(dirname ${exp_log_file})/${exp_framework_id}_${exp_device}_${exp_model}.batch"
is_batch_good "${__batch_file__}" "${exp_device_batch}" || exit 0
echo "__exp.framework_title__= \"${__framework__}\"" >> ${exp_log_file}
# This script is to be executed inside docker container or on a host machine.
# Thus, the environment must be initialized inside this scrip lazily.
[ -z "${runtime_limit_resources}" ] && runtime_limit_resources=":;"
script="\
    export ${tensorflow_env};\
    echo -e \"__tensorflow.version__= \x22\$(python -c 'import tensorflow as tf; print tf.__version__;')\x22\";\
    echo -e \"__results.start_time__= \x22\$(date +%Y-%m-%d:%H:%M:%S:%3N)\x22\";\
    echo ${tensorflow_env}; \
    ${runtime_limit_resources}\
    ${runtime_bind_proc} python ${tensorflow_python_path}/tf_cnn_benchmarks.py ${tensorflow_args} &\
    proc_pid=\$!;\
    [ \"${monitor_frequency}\" != \"0\" ] && echo -e \"\${proc_pid}\" > ${monitor_backend_pid_folder}/proc.pid;\
    wait \${proc_pid};\
    echo -e \"__results.end_time__= \x22\$(date +%Y-%m-%d:%H:%M:%S:%3N)\x22\";\
    echo -e \"__results.proc_pid__= \${proc_pid}\";\
"
if [ "${exp_env}" = "docker" ]; then
    assert_docker_img_exists ${tensorflow_docker_image}
    ${exp_docker_launcher} run ${tensorflow_docker_args} /bin/bash -c "eval $script" >> ${exp_log_file} 2>&1
else
    eval $script >> ${exp_log_file} 2>&1
fi

# Do some post-processing
if tf_error ${exp_log_file}; then
    logwarn "error in \"${exp_log_file}\" with effective batch ${exp_effective_batch} (per device batch ${exp_device_batch})";
    update_error_file "${__batch_file__}" "${exp_device_batch}";
else
    # If everything's OK, we now can get IPS (throughput):
    # What needs be done: check for error
    ips=$(grep "total images/sec:" ${exp_log_file}  | awk '{print $3}')
    if (( $(bc <<< "$ips > 0") )); then
      training_tm=$(echo "${exp_effective_batch} * 1000.0 / $ips" | bc -l)
      echo -e "__results.${exp_phase}_time__= ${training_tm}" >> ${exp_log_file}
      echo -e "__results.${exp_phase}_time_mean__= ${training_tm}" >> ${exp_log_file}
    fi
fi
