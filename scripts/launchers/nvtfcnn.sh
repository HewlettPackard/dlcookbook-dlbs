#!/bin/bash
unknown_params_action=set
. $DLBS_ROOT/scripts/parse_options.sh || exit 1;    # Parse command line options
. $DLBS_ROOT/scripts/utils.sh
loginfo "$0 $*" >> ${exp_log_file}                  # Log command line arguments for debugging purposes
# Now, we have support only for training
if [ "${exp_phase}" == "inference" ]; then
  report_and_exit "failure" "NVTFCNN TensorFlow can only benchmark 'training' phase." "${exp_log_file}";
fi
echo "__exp.framework_title__=\"TensorFlow-nvtfcnn\"" >> ${exp_log_file}
# Check batch is small enough for this experiment
__batch_file__="$(dirname ${exp_log_file})/${exp_framework}_nvtfcnn_${exp_device_type}_${exp_model}.batch"
is_batch_good "${__batch_file__}" "${exp_replica_batch}" || {
  report_and_exit "skipped" "The replica batch size (${exp_replica_batch}) is too large for given SW/HW configuration." "${exp_log_file}";
}
[ "${exp_docker}" = "false" ] && logfatal "NVTFCNN only runs in docker container"
# A hack to make it work with CPUs
update_cmd=":;"
if [ "${exp_device_type}" = "cpu" ]; then
    update_cmd="sed -i '224s/config/#config/' ./nvutils/runner.py && sed -i '225s/config/#config/' ./nvutils/runner.py"
    update_cmd="${update_cmd} && sed -i '90s/gpu/cpu/' ./nvutils/runner.py"
    update_cmd="${update_cmd} && sed -i 's/channels_first/channels_last/' ./${nvtfcnn_model_file}.py"
fi
# This script is to be executed inside docker container or on a host machine.
# Thus, the environment must be initialized inside this scrip lazily.
[ -z "${runtime_launcher}" ] && runtime_launcher=":;"
script="\
    export ${nvtfcnn_env};\
    echo -e \"__exp.framework_ver__= \x22\$(python -c 'import tensorflow as tf; print (tf.__version__);')\x22\";\
    echo -e \"__results.start_time__= \x22\$(date +%Y-%m-%d:%H:%M:%S:%3N)\x22\";\
    cd /workspace/nvidia-examples/cnn && \
    ${update_cmd} && \
    mpiexec ${nvtfcnn_mpi_args} python ${nvtfcnn_model_file}.py ${nvtfcnn_args} &\
    proc_pid=\$!;\
    [ \"${monitor_frequency}\" != \"0\" ] && echo -e \"\${proc_pid}\" > ${monitor_backend_pid_folder}/proc.pid;\
    wait \${proc_pid};\
    echo -e \"__results.end_time__= \x22\$(date +%Y-%m-%d:%H:%M:%S:%3N)\x22\";\
    echo -e \"__results.proc_pid__= \${proc_pid}\";\
"

assert_docker_img_exists ${exp_docker_image}
${exp_docker_launcher} run ${nvtfcnn_docker_args} /bin/bash -c "eval $script" >> ${exp_log_file} 2>&1


${runtime_python} ${nvtfcnn_python_path}/postprocess.py "${exp_log_file}" ${exp_model} ${exp_effective_batch} >> ${exp_log_file} 2>&1
