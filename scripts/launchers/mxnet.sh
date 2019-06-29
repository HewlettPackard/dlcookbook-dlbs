#!/bin/bash
unknown_params_action=set
. ${DLBS_ROOT}/scripts/parse_options.sh || exit 1;    # Parse command line options
. ${DLBS_ROOT}/scripts/utils.sh
loginfo "$0 $*" >> ${exp_log_file}                  # Log command line arguments for debugging purposes
if [[ "${exp_status}" = "simulate" ]]; then
    echo "${mxnet_env} ${runtime_launcher} python ${mxnet_bench_path}/mxnet_benchmarks/benchmarks.py ${mxnet_args}"
    exit 0
fi
# Check batch is small enough for this experiment
__batch_file__="$(dirname ${exp_log_file})/${exp_framework}_${exp_device_type}_${exp_model}.batch"
if [[ "${exp_ignore_past_errors}" != "true" ]]; then
  is_batch_good "${__batch_file__}" "${exp_replica_batch}" || {
    report_and_exit "skipped" "The replica batch size (${exp_replica_batch}) is too large for given SW/HW configuration." "${exp_log_file}";
  }
fi
# This script is to be executed inside docker container or on a host machine.
# Thus, the environment must be initialized inside this scrip lazily.
[[ -z "${runtime_launcher}" ]] && runtime_launcher=":;"

if [[ "${mxnet_kv_store}" == "horovod" ]]; then
  mxnet_exec="${runtime_launcher} mpiexec  --allow-run-as-root --bind-to none --map-by slot -np ${exp_num_gpus} --mca pml ob1 --mca btl ^openib  "
  bench_launcher=""
elif [[ "${exp_num_nodes}" == "1" ]]; then
  mxnet_exec="${runtime_launcher} "
  bench_launcher=""
else
  mxnet_exec="${runtime_launcher} "
  bench_launcher="${mxnet_bench_path}/mxnet_benchmarks/cluster_launcher.py"
  bench_launcher="${bench_launcher} --rendezvous=${mxnet_rendezvous} --num_workers=${exp_num_nodes}"
  bench_launcher="${bench_launcher} --scheduler=${mxnet_scheduler}"
fi
echo "MXNET exec: ${mxnet_exec}" >> ${exp_log_file}
echo "BENCH exec: ${bench_launcher}" >> ${exp_log_file}


script="\
    export ${mxnet_env};\
    echo -e \"__results.start_time__= \x22\$(date +%Y-%m-%d:%H:%M:%S:%3N)\x22\";\
    ${mxnet_exec} ${runtime_python} ${bench_launcher} ${mxnet_bench_path}/mxnet_benchmarks/benchmarks.py ${mxnet_args} &\
    proc_pid=\$!;\
    [[ \"${monitor_frequency}\" != \"0\" ]] && echo -e \"\${proc_pid}\" > ${monitor_backend_pid_folder}/proc.pid;\
    wait \${proc_pid};\
    echo -e \"__results.end_time__= \x22\$(date +%Y-%m-%d:%H:%M:%S:%3N)\x22\";\
    echo -e \"__results.proc_pid__= \${proc_pid}\";\
"
if [[ "${exp_docker}" = "true" ]]; then
    assert_docker_img_exists "${exp_docker_image}" "${exp_docker_launcher}"
    ${exp_docker_launcher} run ${mxnet_docker_args} /bin/bash -c "eval ${script}" >> ${exp_log_file} 2>&1
else
    eval ${script} >> ${exp_log_file} 2>&1
fi

if mxnet_error ${exp_log_file} ${exp_phase}; then
    logwarn "error in \"${exp_log_file}\" with effective batch ${exp_effective_batch} (replica batch ${exp_replica_batch})";
    update_error_file "${__batch_file__}" "${exp_replica_batch}";
    echo "__exp.status__=\"failure\"" >> ${exp_log_file}
fi
