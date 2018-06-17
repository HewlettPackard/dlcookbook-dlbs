#!/bin/bash
unknown_params_action=set
. $DLBS_ROOT/scripts/parse_options.sh || exit 1;    # Parse command line options
. $DLBS_ROOT/scripts/utils.sh
loginfo "$0 $*" >> ${exp_log_file}                  # Log command line arguments for debugging purposes
# Now, we have support only for training
if [ "${exp_phase}" == "inference" ]; then
  report_and_exit "failure" "NVCNN TensorFlow can only benchmark 'training' phase." "${exp_log_file}";
fi
echo "__exp.framework_title__=\"TensorFlow-nvcnn-hvd\"" >> ${exp_log_file}
if [ "$exp_status" = "simulate" ]; then
    echo "${nvcnn_hvd_env} ${runtime_launcher} python ${nvcnn_hvd_python_path}/nvcnn.py ${nvcnn_hvd_args}"
    exit 0
fi
# Check batch is small enough for this experiment
__batch_file__="$(dirname ${exp_log_file})/${exp_framework}_nvcnn_hvd_${exp_device_type}_${exp_model}.batch"
is_batch_good "${__batch_file__}" "${exp_replica_batch}" || {
  report_and_exit "skipped" "The replica batch size (${exp_replica_batch}) is too large for given SW/HW configuration." "${exp_log_file}";
}
# This script is to be executed inside Singularity container or on a host machine.
# Thus, the environment must be initialized inside this scrip lazily.
[ -z "${runtime_launcher}" ] && runtime_launcher=":;"
assert_not_docker_and_singularity
if [ "${exp_singularity}" = "true" ];then
	f=$(mktemp -p ${runtime_multinode_shared_dir})
    d=$(cd $(dirname $f);pwd)
    s=$(basename $f)
    echo "#!/bin/bash" >$f
    echo "function finish {" >> $f
    echo "rm -f $f" >> $f
    echo "}" >> $f
    echo "trap finish EXIT" >> $f
    echo "export ${nvcnn_hvd_env}" >> $f
    echo "echo -e \x22__exp.framework_ver__= \x24\x28python -c \x27import tensorflow as tf; print \x28tf.__version__\x29;\x27\x29\x22" >> $f
    echo "echo -e \x22__results.start_time__= \x24\x28date +%Y-%m-%d:%H:%M:%S:%3N\\x29\x22" >> $f
    echo "${runtime_launcher} ${runtime_python} -u ${nvcnn_hvd_python_path}/nvcnn_hvd.py ${nvcnn_hvd_args} & proc_pid=\x24!; \\" >> $f
    echo "[ \x22${monitor_frequency}\x22 != \x220\x22 ] && echo -e \x22\x24{proc_pid}\x22 > ${monitor_pid_folder}/proc.pid" >> $f
    echo "wait \x24{proc_pid}" >> $f
    echo "echo -e \x22__results.end_time__= \x24\x28date +%Y-%m-%d:%H:%M:%S:%3N\x29\x22" >> $f
    echo "echo -e \x22__results.proc_pid__= \x24{proc_pid}\x22" >> $f
    sed -i -e 's:\\x27:'"'"':g' -e 's:\\x22:":g' -e 's:\\x28:(:g' -e 's:\\x29:):g' -e 's:\\x24:$:g' $f
    #assert_singularity_img_exists ${exp_singularity_image}
  
    nvcnn_hvd_singularity_args_temp="-B ${d}:/workspace/temp ${nvcnn_hvd_singularity_args}"
	echo "${nvcnn_hvd_mpirun} -H ${nvcnn_hvd_mpirun_hosts} -np ${nvcnn_hvd_mpirun_num_tasks} ${nvcnn_hvd_mpirun_args} ${exp_singularity_launcher} exec ${nvcnn_hvd_singularity_args_temp}"
	${nvcnn_hvd_mpirun} -H ${nvcnn_hvd_mpirun_hosts} -np ${nvcnn_hvd_mpirun_num_tasks} ${nvcnn_hvd_mpirun_args} ${exp_singularity_launcher} exec ${nvcnn_hvd_singularity_args_temp} /bin/bash /workspace/temp/${s} >> ${exp_log_file} 2>&1
    rm -f $f
else
    echo "bare metal hostname $(hostname)"
    script="\
        export ${nvcnn_hvd_env};\
        echo -e \"__exp.framework_ver__= \x22\$(python -c 'import tensorflow as tf; print (tf.__version__);')\x22\";\
        echo -e \"__results.start_time__= \x22\$(date +%Y-%m-%d:%H:%M:%S:%3N)\x22\";\
        echo "bare metal hostname $(hostname)";\
        ${runtime_launcher} ${runtime_python} ${nvcnn_hvd_python_path}/nvcnn.py ${nvcnn_hvd_args} &\
        proc_pid=\$!;\
        [ \"${monitor_frequency}\" != \"0\" ] && echo -e \"\${proc_pid}\" > ${monitor_backend_pid_folder}/proc.pid;\
        wait \${proc_pid};\
        echo -e \"__results.end_time__= \x22\$(date +%Y-%m-%d:%H:%M:%S:%3N)\x22\";\
        echo -e \"__results.proc_pid__= \${proc_pid}\";\
    "
    #Assume bare metal if not Singularity
    eval $script >> ${exp_log_file} 2>&1
fi

# Do some post-processing
if tf_error ${exp_log_file}; then
    logwarn "error in \"${exp_log_file}\" with effective batch ${exp_effective_batch} (replica batch ${exp_replica_batch})";
    update_error_file "${__batch_file__}" "${exp_replica_batch}";
    echo "__exp.status__=\"failure\"" >> ${exp_log_file}
else
    :;
    # If everything's OK, we now can get IPS (throughput):
    # What needs be done: check for error
    #ips=$(grep "Images/sec:" ${exp_log_file}  | awk '{print $2}')
    #is_positive=$(echo "print($ips > 0)" | python)
    #if [ "$is_positive" == "True" ]; then
    #  tm=$(echo "print (${exp_effective_batch} * 1000.0 / $ips)" | python)
    #  echo -e "__results.time__= $tm" >> ${exp_log_file}
    #  echo -e "__results.throughput__= $ips" >> ${exp_log_file}
    #fi
fi
