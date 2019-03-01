#!/bin/bash
unknown_params_action=set
. $DLBS_ROOT/scripts/parse_options.sh || exit 1;    # Parse command line options
. $DLBS_ROOT/scripts/utils.sh
loginfo "$0 $*" >> ${exp_log_file}                  # Log command line arguments for debugging purposes
echo "__exp.framework_title__=\"TensorFlow\"" >> ${exp_log_file}
if [[ "$exp_status" = "simulate" ]]; then
    echo "${tensorflow_env} ${runtime_launcher} python ${tensorflow_python_path}/tf_cnn_benchmarks.py ${tensorflow_args}"
    exit 0
fi
# Check batch is small enough for this experiment if this is not disabled.
 __batch_file__="$(dirname ${exp_log_file})/${exp_framework}_${exp_device_type}_${exp_model}.batch"
if [[ "${exp_ignore_past_errors}" != "true" ]]; then
  is_batch_good "${__batch_file__}" "${exp_replica_batch}" || {
    report_and_exit "skipped" "The replica batch size (${exp_replica_batch}) is too large for given SW/HW configuration." "${exp_log_file}";
  }
fi
# Check if we have a required TF_CNN_BENCHMARK version. This is a temporary solution for now.
if [[ -z "${tensorflow_git_hashtag}" ]]; then
    proj_dir="${DLBS_ROOT}/python/tf_cnn_benchmarks${tensorflow_git_hashtag}"
    if [[ ! -d "${proj_dir}" ]]; then
        # There's no directory, so, I assume DLBS never cloned it.
        mkdir -p "${proj_dir}"
        tmp_dir=${proj_dir}/tmpd7a7052a
        git clone https://github.com/tensorflow/benchmarks.git ${tmp_dir}
        cp -R ${tmp_dir}/scripts/tf_cnn_benchmarks/* ${proj_dir}
        rm -rf ${tmp_dir}
    fi
fi
# This script is to be executed inside docker container or on a host machine.
# Thus, the environment must be initialized inside this scrip lazily.
[[ -z "${runtime_launcher}" ]] && runtime_launcher=":;"
script="\
    export ${tensorflow_env};\
    echo -e \"__exp.framework_ver__= \x22\$(${runtime_python} -c 'import tensorflow as tf; print(tf.__version__);' | tr -d '\n')\x22\";\
    echo -e \"__results.start_time__= \x22\$(date +%Y-%m-%d:%H:%M:%S:%3N)\x22\";\
    ${runtime_launcher} ${runtime_python} ${tensorflow_python_path}/tf_cnn_benchmarks.py ${tensorflow_args} &\
    proc_pid=\$!;\
    [ \"${monitor_frequency}\" != \"0\" ] && echo -e \"\${proc_pid}\" > ${monitor_backend_pid_folder}/proc.pid;\
    wait \${proc_pid};\
    echo -e \"__results.end_time__= \x22\$(date +%Y-%m-%d:%H:%M:%S:%3N)\x22\";\
    echo -e \"__results.proc_pid__= \${proc_pid}\";\
"
if [[ "${exp_docker}" = "true" ]]; then
    assert_docker_img_exists "${exp_docker_image}" "${exp_docker_launcher}"
    ${exp_docker_launcher} run ${tensorflow_docker_args} /bin/bash -c "eval $script" >> ${exp_log_file} 2>&1
else
    eval ${script} >> ${exp_log_file} 2>&1
fi

python ${DLBS_ROOT}/python/dlbs/logger.py "tf_cnn_benchmarks" "${exp_log_file}"
