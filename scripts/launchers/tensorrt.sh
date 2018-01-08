#!/bin/bash
unknown_params_action=set
. $DLBS_ROOT/scripts/parse_options.sh || exit 1;    # Parse command line options
. $DLBS_ROOT/scripts/utils.sh
loginfo "$0 $*"  >> ${exp_log_file}                 # Log command line arguments for debugging purposes
# The simulation mode: just print out what is about to be launched
if [ "$exp_simulation" = "true" ]; then
    echo "${runtime_bind_proc} tensorrt ${tensorrt_args}"
    exit 0
fi

# Check batch is small enough for this experiment
__batch_file__="$(dirname ${exp_log_file})/${exp_framework_id}_${exp_device}_${exp_model}.batch"
is_batch_good "${__batch_file__}" "${exp_device_batch}" || { logwarn "Device batch is too big for configuration"; exit 0; }

# Make sure model exists
host_model_dir=$DLBS_ROOT/models/${exp_model}
model_file=$(find ${host_model_dir}/ -name "*.${exp_phase}.prototxt")
file_exists "$model_file" || { logerr "model file does not exist"; exit 1; }

# Copy model file and replace batch size there.


remove_files "${host_model_dir}/${caffe_model_file}"
cp ${model_file} ${host_model_dir}/${caffe_model_file} || logfatal "Cannot cp \"${model_file}\" to \"${host_model_dir}/${caffe_model_file}\""

sed -i "s/__EXP_DEVICE_BATCH__/${exp_device_batch}/g" ${host_model_dir}/${caffe_model_file}
net_name=$(get_value_by_key "${host_model_dir}/${caffe_model_file}" "name")

echo "__exp.model_title__= \"${net_name}\"" >> ${exp_log_file}
echo "__exp.framework_title__= \"TensorRT\"" >> ${exp_log_file}

# This script is to be executed inside docker container or on a host machine.
# Thus, the environment must be initialized inside this scrip lazily.
[ -z "${runtime_limit_resources}" ] && runtime_limit_resources=":;"
echo "runtime_limit_resources: \"${runtime_limit_resources}\""
script="\
    export ${tensorrt_env};\
    echo -e \"__tensorrt.version__= \x22\$(tensorrt --version)\x22\";\
    echo -e \"__results.start_time__= \x22\$(date +%Y-%m-%d:%H:%M:%S:%3N)\x22\";\
    ${runtime_limit_resources}\
    ${runtime_bind_proc} tensorrt ${tensorrt_args} &\
    proc_pid=\$!;\
    [ \"${monitor_frequency}\" != \"0\" ] && echo -e \"\${proc_pid}\" > ${monitor_backend_pid_folder}/proc.pid;\
    wait \${proc_pid};\
    echo -e \"__results.end_time__= \x22\$(date +%Y-%m-%d:%H:%M:%S:%3N)\x22\";\
    echo -e \"__results.proc_pid__= \x22\${proc_pid}\x22\";\
"

if [ "${exp_env}" = "docker" ]; then
    assert_docker_img_exists ${tensorrt_docker_image}
    ${exp_docker_launcher} run ${tensorrt_docker_args} /bin/bash -c "eval $script" >> ${exp_log_file} 2>&1
else
    eval $script >> ${exp_log_file} 2>&1
fi

remove_files "${host_model_dir}/${caffe_model_file}"
