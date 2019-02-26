#!/bin/bash
unknown_params_action=set
. $DLBS_ROOT/scripts/parse_options.sh || exit 1;    # Parse command line options
. $DLBS_ROOT/scripts/utils.sh
loginfo "$0 $*" >> ${exp_log_file}                  # Log command line arguments for debugging purposes
# The simulation mode: just print out what is about to be launched
if [ "$exp_status" = "simulate" ]; then
    echo "${tensorrt_env} ${runtime_launcher} tensorrt ${tensorrt_args}"
    exit 0
fi
[ "${exp_phase}" == "training" ] && \
    report_and_exit "failure" "TensorRT benchmark backend does not support 'training' phase (exp.phase=training)." "${exp_log_file}"
# Do model checking and preparation only if not fake inference.
if [ "${tensorrt_fake_inference}" == "false" ]; then
    # Check batch is small enough for this experiment
    __batch_file__="$(dirname ${exp_log_file})/${exp_framework}_${exp_device_type}_${exp_model}.batch"
    is_batch_good "${__batch_file__}" "${exp_replica_batch}" || {
        report_and_exit "skipped" "The replica batch size (${exp_replica_batch}) is too large for given SW/HW configuration." "${exp_log_file}";
    }
    # Make sure model exists
    if [ "${tensorrt_user_model}" == "false" ]; then
        host_model_dir=$DLBS_ROOT/models/${exp_model}
        model_file=$(find ${host_model_dir}/ -name "*.${exp_phase}.prototxt")
        file_exists "$model_file" || \
            report_and_exit "failure" "A model file (${host_model_dir}/*.${exp_phase}.prototxt) does not exist." "${exp_log_file}"

        # Copy model file and replace batch size there.
        remove_files "${host_model_dir}/${caffe_model_file}"
        cp ${model_file} ${host_model_dir}/${caffe_model_file} || {
            report_and_exit "failure" "Cannot copy \"${model_file}\" to \"${host_model_dir}/${caffe_model_file}\"" "${exp_log_file}"
        }

        sed -i "s/__EXP_DEVICE_BATCH__/${exp_replica_batch}/g" ${host_model_dir}/${caffe_model_file}
        net_name=$(get_value_by_key "${host_model_dir}/${caffe_model_file}" "name")
        echo "__exp.model_title__= \"${net_name}\"" >> ${exp_log_file}
    fi
fi
# If does not exist, create calibration cache path
[ ! -z ${tensorrt_cache} ] && mkdir -p ${tensorrt_cache}
# This script is to be executed inside docker container or on a host machine.
# Thus, the environment must be initialized inside this scrip lazily.
[ -z "${runtime_launcher}" ] && runtime_launcher=":;"
script="          export ${tensorrt_env}"
script="$script;"'echo -e "__exp.framework_ver__= \x22$(tensorrt --version)\x22"'
script="$script;"'echo -e "__results.start_time__= \x22$(date +%Y-%m-%d:%H:%M:%S:%3N)\x22"'
script="$script;${runtime_launcher} tensorrt ${tensorrt_args}"'& proc_pid=$!'
script="$script; [ \x22${monitor_frequency}\x22 != \x220\x22 ] && "'echo -e "${proc_pid}"'"> ${monitor_backend_pid_folder}/proc.pid"
script="$script;"'wait ${proc_pid}'
script="$script;"'echo -e "__results.end_time__= \x22$(date +%Y-%m-%d:%H:%M:%S:%3N)\x22"'
script="$script;"'echo -e "__results.proc_pid__= ${proc_pid}";'

if [ "${exp_docker}" = "true" ]; then
    assert_docker_img_exists "${exp_docker_image}" "${exp_docker_launcher}"
    ${exp_docker_launcher} run ${tensorrt_docker_args} /bin/bash -c "eval $script" >> ${exp_log_file} 2>&1
else
    eval $script >> ${exp_log_file} 2>&1
fi

if [ "${tensorrt_fake_inference}" == "false" ]; then
    remove_files "${host_model_dir}/${caffe_model_file}"
fi
