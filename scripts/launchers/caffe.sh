#!/bin/bash

unknown_params_action=set
. $DLBS_ROOT/scripts/parse_options.sh || exit 1;    # Parse command line options
. $DLBS_ROOT/scripts/utils.sh
loginfo "$0 $*" >> ${exp_log_file}                  # Log command line arguments for debugging purposes
echo command line
echo $@
exit
# The simulation mode: just print out what is about to be launched
if [ "$exp_status" = "simulate" ]; then
  echo "${caffe_env} ${runtime_launcher} caffe ${caffe_action} ${caffe_args}"
  exit 0
fi
# Check batch is small enough for this experiment
__batch_file__="$(dirname ${exp_log_file})/${exp_framework}_${exp_device_type}_${exp_model}.batch"
is_batch_good "${__batch_file__}" "${exp_replica_batch}" || {
  report_and_exit "skipped" "The replica batch size (${exp_replica_batch}) is too large for given SW/HW configuration." "${exp_log_file}";
}
# Make sure model exists
host_model_dir=$DLBS_ROOT/python/caffe_benchmarks/models/${exp_model}
model_file=$(find ${host_model_dir}/ -name "*.${exp_phase}.prototxt")
file_exists "$model_file" || report_and_exit "failure" "A model file ($model_file) does not exist." "${exp_log_file}"
# Copy model file and replace batch size there.
# https://github.com/BVLC/caffe/blob/master/docs/multigpu.md
# NOTE: each GPU runs the batchsize specified in your train_val.prototxt
remove_files "${host_model_dir}/${caffe_model_file}" "${host_model_dir}/${caffe_solver_file}"
cp ${model_file} ${host_model_dir}/${caffe_model_file} || {
  report_and_exit "failure" "Cannot copy \"${model_file}\" to \"${host_model_dir}/${caffe_model_file}\"" "${exp_log_file}"
}
# If we are in 'training' phase and data_dir is not empty, we need to change *.train.prototxt file here.
# Or we can have two configurations for synthetic/real data.
# Or we can specify input layers in JSON config, so that we can basically set this dynamically
if [ "${exp_phase}" == "training" ]; then
    if [ "${exp_data_dir}" == "" ]; then
        sed -i "s/^#synthetic//g" ${host_model_dir}/${caffe_model_file}
    else
        if [ "${exp_docker}" == "true" ]; then
            real_data_dir="/workspace/data"
            real_data_mean_file="/workspace/image_mean/${caffe_data_mean_file_name}"
        else
            real_data_dir="${exp_data_dir}"
            real_data_mean_file="${caffe_data_mean_file}"
        fi
        sed -i "s/^#data//g" ${host_model_dir}/${caffe_model_file}
        sed -i "s#__CAFFE_MIRROR__#${caffe_mirror}#g" ${host_model_dir}/${caffe_model_file}
        sed -i "s#__CAFFE_DATA_MEAN_FILE__#${real_data_mean_file}#g" ${host_model_dir}/${caffe_model_file}
        sed -i "s#__CAFFE_DATA_DIR__#${real_data_dir}#g" ${host_model_dir}/${caffe_model_file}
        sed -i "s#__CAFFE_DATA_BACKEND__#${caffe_data_backend}#g" ${host_model_dir}/${caffe_model_file}
    fi
    if [ "${exp_framework_fork}" == "nvidia" ]; then
        sed -i "s/^#precision//g" ${host_model_dir}/${caffe_model_file}
        sed -i "s/__FORWARD_TYPE___/${caffe_nvidia_forward_precision}/g" ${host_model_dir}/${caffe_model_file}
        sed -i "s/__BACKWARD_TYPE___/${caffe_nvidia_backward_precision}/g" ${host_model_dir}/${caffe_model_file}
        sed -i "s/__FORWARD_MATH___/${caffe_nvidia_forward_math_precision}/g" ${host_model_dir}/${caffe_model_file}
        sed -i "s/__BACKWARD_MATH___/${caffe_nvidia_backward_math_precision}/g" ${host_model_dir}/${caffe_model_file}
    fi
fi
# BVLC/Intel Caffe and NVIDIA Caffe treat batch sizes in protobuf files differently.
# BVLC/Intel Caffe work in weak scaling mode. Batch size in protobuf is a per-device batch size.
# NVIDIA Caffe on the other hand treat the value for batch size as effective batch size dividing it
# by number of solvers (workers). However, because of the algorithm which is here:
# https://github.com/NVIDIA/caffe/blob/cdb3d9a5d46774a3be3cc4c4ecc0bcd760901cc1/src/caffe/parallel.cpp#L279
# in some cases NVIDIA Caffe can treat batch size as per-GPU batch size switching to weak scaling mode.
# This is exactly our case with synthetic data where layer of type Input is not processed in that function
# resulting in batch size not being modified.
if [ "${exp_framework_fork}" == "nvidia" ]; then
    # NVIDIA Caffe - strong scaling for real data and weak scaling for synthetic one
    if [ "${exp_data_dir}" == "" ]; then
        # Synthetic data with 'Input' layer - Caffe is in weak scaling model
        sed -i "s/__EXP_DEVICE_BATCH__/${exp_replica_batch}/g" ${host_model_dir}/${caffe_model_file}
    else
        # Real data - Caffe is in strong scaling mode - it will divide whatever batch size we have in
        # protobuf by number of solvers.
        sed -i "s/__EXP_DEVICE_BATCH__/${exp_effective_batch}/g" ${host_model_dir}/${caffe_model_file}
    fi
else
    # This is for BVLC/Intel Caffe
    sed -i "s/__EXP_DEVICE_BATCH__/${exp_replica_batch}/g" ${host_model_dir}/${caffe_model_file}
fi
#
net_name=$(get_value_by_key "${host_model_dir}/${caffe_model_file}" "name")
[ "${exp_phase}" = "training" ] && echo -e "${caffe_solver}" > ${host_model_dir}/${caffe_solver_file}
echo "__exp.model_title__= \"${net_name}\"" >> ${exp_log_file}
# This script is to be executed inside docker container or on a host machine.
# Thus, the environment must be initialized inside this scrip lazily.
[ -z "${runtime_launcher}" ] && runtime_launcher=":;"
script="\
    export ${caffe_env};\
    echo -e \"__exp.framework_ver__= \x22\$(caffe --version | head -1 | awk '{print \$3}')\x22\";\
    echo -e \"__results.start_time__= \x22\$(date +%Y-%m-%d:%H:%M:%S:%3N)\x22\";\
    ${runtime_launcher} caffe ${caffe_action} ${caffe_args} &\
    proc_pid=\$!;\
    [ \"${monitor_frequency}\" != \"0\" ] && echo -e \"\${proc_pid}\" > ${monitor_backend_pid_folder}/proc.pid;\
    wait \${proc_pid};\
    echo -e \"__results.end_time__= \x22\$(date +%Y-%m-%d:%H:%M:%S:%3N)\x22\";\
    echo -e \"__results.proc_pid__= \${proc_pid}\";\
"

if [ "${exp_docker}" = "true" ]; then
    assert_docker_img_exists ${exp_docker_image}
    ${exp_docker_launcher} run ${caffe_docker_args} /bin/bash -c "eval $script" >> ${exp_log_file} 2>&1
else
    eval $script >> ${exp_log_file} 2>&1
fi

# Do some post-processing
remove_files "${host_model_dir}/${caffe_model_file}" "${host_model_dir}/${caffe_solver_file}"
caffe_postprocess_log "${exp_log_file}" "${__batch_file__}" "${exp_phase}" "${exp_replica_batch}" "${exp_effective_batch}" "${exp_num_batches}"
