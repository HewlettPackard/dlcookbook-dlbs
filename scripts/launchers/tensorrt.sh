#!/bin/bash
loginfo "$0 $*"  >> ${exp_log_file}                 # Log command line arguments for debugging purposes
unknown_params_action=set
. $DLBS_ROOT/scripts/parse_options.sh || exit 1;    # Parse command line options
. $DLBS_ROOT/scripts/utils.sh

# Check batch is small enough for this experiment
__batch_file__="$(dirname ${exp_log_file})/${exp_framework_id}_${exp_device}_${exp_model}.batch"
is_batch_good "${__batch_file__}" "${exp_device_batch}" || { logwarn "Batch too large."; exit 0; }

# Make sure model exists
host_model_dir=$DLBS_ROOT/models/${exp_model}
host_def_file=$(find ${host_model_dir}/ -name "*.deploy.prototxt")
file_exists "$host_def_file" || { logerr "Host def file does not exist"; exit 1; }
# Make sure corresponding *.caffemodel exists.
host_model_file="${host_def_file%.deploy.prototxt}.caffemodel"
file_exists "$host_model_file" || { logerr "Host model file does not exist"; exit 1; }
# Find file with model properties. It must exist.
opts_file="${host_model_dir}/opts"
[ -f "$opts_file" ] || { logwarn "Opt file ($opts_file) does not exist"; exit 1; }
net_name=$(get_value_by_key "${host_model_dir}/${caffe_model_file}" "name")
int_infer=$(get_value_by_key $opts_file "int8_infer") || logfatal "$int_infer";
# Some models cannot be used with INT8
[ "${int_infer}" = "no" ] && [ "${tensorrt_precision}" = "int8" ] && { logwarn "INT8 is not supported for this model"; exit 1; }
# Change batch size
remove_files "${host_model_dir}/${tensorrt_def_file}"
cp ${host_def_file} ${host_model_dir}/${tensorrt_def_file} || logfatal "Cannot cp \"${host_def_file}\" to \"${host_model_dir}/${tensorrt_def_file}\""
sed -i "s/__BATCH_SIZE__/${exp_device_batch}/g" ${host_model_dir}/${tensorrt_def_file}
#
input_size=$(get_value_by_key $opts_file "input_size") || logfatal "$input_size";
trt_args="/workspace/model/${tensorrt_def_file} /workspace/model/$(basename ${host_model_file}) ${tensorrt_precision} ${input_size} ${exp_device_batch} ${exp_bench_iters} data prob"
#
if [ "$exp_simulation" = "true" ]; then
    echo "${runtime_bind_proc} tensorrt ${trt_args}"
    exit 0
fi
echo "__exp.model_title__= \"${net_name}\"" >> ${exp_log_file}
echo "__exp.framework_title__= \"TensorRT\"" >> ${exp_log_file}
[ -z "${runtime_limit_resources}" ] && runtime_limit_resources=":;"
echo "runtime_limit_resources: \"${runtime_limit_resources}\""
script="\
    export ${tensorrt_env};\
    echo -e \"__results.start_time__= \x22\$(date +%Y-%m-%d:%H:%M:%S:%3N)\x22\";\
    ${runtime_limit_resources}\
    ${runtime_bind_proc} tensorrt ${trt_args} &\
    proc_pid=\$!;\
    [ "${resource_monitor_enabled}" = "true" ] && echo -e \"\${proc_pid}\" > ${resource_monitor_pid_file_folder}/proc.pid;\
    wait \${proc_pid};\
    echo -e \"__results.end_time__= \x22\$(date +%Y-%m-%d:%H:%M:%S:%3N)\x22\";\
    echo -e \"__results.proc_pid__= \x22\${proc_pid}\x22\";\
"
echo $script
assert_docker_img_exists ${tensorrt_docker_image}
${exp_docker_launcher} run ${tensorrt_docker_args} /bin/bash -c "eval $script" >> ${exp_log_file} 2>&1

remove_files "${host_model_dir}/${tensorrt_def_file}"
