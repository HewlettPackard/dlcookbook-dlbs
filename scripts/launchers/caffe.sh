#!/bin/bash
unknown_params_action=set
. $DLBS_ROOT/scripts/parse_options.sh || exit 1;    # Parse command line options
. $DLBS_ROOT/scripts/utils.sh
loginfo "$0 $*" >> ${exp_log_file}                  # Log command line arguments for debugging purposes
# The simulation mode: just print out what is about to be launched
if [ "$exp_simulation" = "true" ]; then
    echo "${runtime_bind_proc} caffe ${caffe_action} ${caffe_args}"
    exit 0
fi
__framework__="$(echo ${caffe_fork} | tr '[:lower:]' '[:upper:]')_Caffe"
# Check batch is small enough for this experiment
__batch_file__="$(dirname ${exp_log_file})/${exp_framework_id}_${exp_device}_${exp_model}.batch"
is_batch_good "${__batch_file__}" "${exp_device_batch}" || { logwarn "Device batch is too big for configuration"; exit 0; }

# Make sure model exists
host_model_dir=$DLBS_ROOT/models/${exp_model}
model_file=$(find ${host_model_dir}/ -name "*.${exp_phase}.prototxt")
file_exists "$model_file" || { logerr "model file does not exist"; exit 1; }

# Copy model file and replace batch size there.
# https://github.com/BVLC/caffe/blob/master/docs/multigpu.md
# NOTE: each GPU runs the batchsize specified in your train_val.prototxt
remove_files "${host_model_dir}/${caffe_model_file}" "${host_model_dir}/${caffe_solver_file}"
cp ${model_file} ${host_model_dir}/${caffe_model_file} || logfatal "Cannot cp \"${model_file}\" to \"${host_model_dir}/${caffe_model_file}\""
# If we are in 'training' phase and data_dir is not empty, we need to change *.train.prototxt file here.
# Or we can have two configurations for synthetic/real data.
# Or we can specify input layers in JSON config, so that we can basically set this dynamically
if [ "${exp_phase}" == "training" ]; then
    if [ "${caffe_data_dir}" == "" ]; then
        sed -i "s/^#synthetic//g" ${host_model_dir}/${caffe_model_file}
    else
        sed -i "s/^#data//g" ${host_model_dir}/${caffe_model_file}
        sed -i "s#__CAFFE_MIRROR__#${caffe_mirror}#g" ${host_model_dir}/${caffe_model_file}
        sed -i "s#__CAFFE_DATA_MEAN_FILE__#${caffe_data_mean_file}#g" ${host_model_dir}/${caffe_model_file}
        sed -i "s#__CAFFE_DATA_DIR__#${caffe_data_dir}#g" ${host_model_dir}/${caffe_model_file}
        sed -i "s#__CAFFE_DATA_BACKEND__#${caffe_data_backend}#g" ${host_model_dir}/${caffe_model_file}
    fi
    if [ "${exp_framework_id}" == "nvidia_caffe" ]; then
        sed -i "s/^#precision//g" ${host_model_dir}/${caffe_model_file}
        sed -i "s/__FORWARD_TYPE___/${nvidia_caffe_forward_precision}/g" ${host_model_dir}/${caffe_model_file}
        sed -i "s/__BACKWARD_TYPE___/${nvidia_caffe_backward_precision}/g" ${host_model_dir}/${caffe_model_file}
        sed -i "s/__FORWARD_MATH___/${nvidia_caffe_forward_math_precision}/g" ${host_model_dir}/${caffe_model_file}
        sed -i "s/__BACKWARD_MATH___/${nvidia_caffe_backward_math_precision}/g" ${host_model_dir}/${caffe_model_file}
    fi
fi
# For NVIDIA Caffe, protobuf must contain effective batch size.
if [ "${caffe_fork}" == "nvidia" ]; then
    sed -i "s/__EXP_DEVICE_BATCH__/${exp_effective_batch}/g" ${host_model_dir}/${caffe_model_file}
else
    sed -i "s/__EXP_DEVICE_BATCH__/${exp_device_batch}/g" ${host_model_dir}/${caffe_model_file}
fi
#
net_name=$(get_value_by_key "${host_model_dir}/${caffe_model_file}" "name")
[ "${exp_phase}" = "training" ] && echo -e "${caffe_solver}" > ${host_model_dir}/${caffe_solver_file}
echo "__exp.model_title__= \"${net_name}\"" >> ${exp_log_file}
echo "__exp.framework_title__= \"${__framework__}\"" >> ${exp_log_file}
# This script is to be executed inside docker container or on a host machine.
# Thus, the environment must be initialized inside this scrip lazily.
[ -z "${runtime_limit_resources}" ] && runtime_limit_resources=":;"
script="\
    export ${caffe_env};\
    echo -e \"__caffe.version__= \x22\$(caffe --version | head -1 | awk '{print \$3}')\x22\";\
    echo -e \"__results.start_time__= \x22\$(date +%Y-%m-%d:%H:%M:%S:%3N)\x22\";\
    ${runtime_limit_resources}\
    ${runtime_bind_proc} caffe ${caffe_action} ${caffe_args} &\
    proc_pid=\$!;\
    [ "${resource_monitor_enabled}" = "true" ] && echo -e \"\${proc_pid}\" > ${resource_monitor_pid_file_folder}/proc.pid;\
    wait \${proc_pid};\
    echo -e \"__results.end_time__= \x22\$(date +%Y-%m-%d:%H:%M:%S:%3N)\x22\";\
    echo -e \"__results.proc_pid__= \${proc_pid}\";\
"

if [ "${exp_env}" = "docker" ]; then
    assert_docker_img_exists ${caffe_docker_image}
    ${exp_docker_launcher} run ${caffe_docker_args} /bin/bash -c "eval $script" >> ${exp_log_file} 2>&1
else
    eval $script >> ${exp_log_file} 2>&1
fi

# Do some post-processing
remove_files "${host_model_dir}/${caffe_model_file}" "${host_model_dir}/${caffe_solver_file}"
caffe_postprocess_log "${exp_log_file}" "${__batch_file__}" "${exp_phase}" "${exp_device_batch}" "${exp_effective_batch}" "${exp_bench_iters}"
