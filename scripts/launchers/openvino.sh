#!/bin/bash
unknown_params_action=set
. ${DLBS_ROOT}/scripts/parse_options.sh || exit 1;    # Parse command line options
. ${DLBS_ROOT}/scripts/utils.sh
loginfo "$0 $*" >> ${exp_log_file}                    # Log command line arguments for debugging purposes
# The simulation mode: just print out what is about to be launched
if [[ "$exp_status" = "simulate" ]]; then
  echo "${caffe_env} ${runtime_launcher} caffe ${caffe_action} ${caffe_args}"
  exit 0
fi

if [[ ! "${exp_docker}" = "true" ]]; then
    report_and_exit "failure" "OpenVINO backend must use docker." "${exp_log_file}";
fi

# Create DLBS cache dir: this needs to go to DLBS core.
DLBS_CACHE="${runtime_dlbs_cache}"
[[ ! -d "${DLBS_CACHE}/openvino/models" ]] && mkdir -p "${DLBS_CACHE}/openvino/models"
[[ ! -d "${DLBS_CACHE}/openvino/cache" ]] && mkdir -p "${DLBS_CACHE}/openvino/cache"

[[ -z "${runtime_launcher}" ]] && runtime_launcher=":;"
# https://docs.openvinotoolkit.org/latest/_inference_engine_samples_benchmark_app_README.html
#  [ ! -f /mnt/openvino/list_topologies.yml ] && [-f ./list_topologies.yml ] && cp ./list_topologies.yml /mnt/openvino; \
script="\
    source \${OPENVINO_DIR}/bin/setupvars.sh;\
    export ${openvino_env};\
    echo -e \"__results.start_time__= \x22\$(date +%Y-%m-%d:%H:%M:%S:%3N)\x22\";\
    cd \${OPENVINO_DIR}/deployment_tools/open_model_zoo/tools/downloader;\
    ${runtime_python} ./downloader.py ${openvino_downloader_args};\
    cd /opt/intel/openvino_benchmark_app/intel64/Release;\
    ${runtime_launcher} ./benchmark_app -m /mnt/openvino/models/intel/${openvino_model_rel_path} -api ${openvino_api} -b ${exp_replica_batch};\
    proc_pid=\$!;\
    [ \"${monitor_frequency}\" != \"0\" ] && echo -e \"\${proc_pid}\" > ${monitor_backend_pid_folder}/proc.pid;\
    wait \${proc_pid};\
    echo -e \"__results.end_time__= \x22\$(date +%Y-%m-%d:%H:%M:%S:%3N)\x22\";\
    echo -e \"__results.proc_pid__= \${proc_pid}\";\
"

assert_docker_img_exists "${exp_docker_image}" "${exp_docker_launcher}"
${exp_docker_launcher} run ${openvino_docker_args} /bin/bash -c "eval $script" >> ${exp_log_file} 2>&1

# A problem here - runtime.python refer to python for benchmarks, but this runs bare metal.
${runtime_python} ${openvino_python_path}/postprocess.py "${exp_log_file}" "${exp_effective_batch}" >> ${exp_log_file} 2>&1
