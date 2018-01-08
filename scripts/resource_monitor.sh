#!/bin/bash

# |-----|------|------memory------|------|------|-------|
# | PID | TIME | VIRT | RES | SHR | %CPU | %MEM | POWER |
# |     |      |      |     |     |      |      |       |
# |-----------------------------------------------------|

# Start as background process, remove PID file in /dev/shm if exists
# In a loop read that file for a PID to monitor
# Read PID and monitor usage
# If PID changes, add me
# Usage: resource_monitor.sh pid_file log_file interval_seconds

# PID file (must be created by parent script)
pid_file=$1
# Stats file
log_file=$2
# Interval in seconds
interval=$3

ipmitool dcmi power reading >> /dev/null 2>&1 && has_ipmi='true' || has_ipmi='false'
nvidia-smi >> /dev/null 2>&1 && has_nvidia_smi='true' || has_nvidia_smi='false'
power=-1
gpus_power=-1
# Main application loop
while true
do
    proc_pid=$(cat $pid_file 2>/dev/null) || {
        #echo "Proc file ($pid_file) does not exist, waiting...";
	       sleep $interval;
	       continue;
    }
    [ "${proc_pid}XXX" == "XXX" ] && {
	       #echo "Proc file ($pid_file) provides no pid, waiting...";
	       sleep $interval;
	       continue;
    }
    [ "${proc_pid}" == "exit" ] && {
	       #echo "Proc file ($pid_file) provides EXIT tag, exiting...";
	       exit 0
    }
    # Get all subprocesses of parent process
    sproc_pids=$(pgrep -P ${proc_pid} -d,)
    if [ "$sproc_pids" == "" ]; then
	       proc_pids=$proc_pid
    else
	       proc_pids=$proc_pid,$sproc_pids
    fi

    # Get current time stamp
    date=$(date +%Y-%m-%d:%H:%M:%S:%3N)

    # Get matrix (top output) for monitoring process and its subprocesses
    stats=$(top -b -p${proc_pids} -n1 | tail -n +8)
    [ "${stats}XXX" == "XXX" ] && {
        #echo "Process PID ($proc_pid) is invalid (process exited?), waiting...";
        continue
    }
    # Sum over all child processes we need mem columns (5, 6, 7), cpu (9), and mem (10)
    stats=$(echo "$stats" | awk '{virt += $5; res += $6; shrd += $7; cpu += $9; mem += $10} END {print virt, res, shrd, cpu, mem}')

    # If ipmitool is available, use it (root rights issue?)
    if [ "$has_ipmi" == "true" ]; then
        power=$(ipmitool dcmi power reading | grep "Instantaneous" | awk '{print $4}') || power='-1'
    fi
    if [ "$has_nvidia_smi" == "true" ]; then
        gpus_power=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits) || gpus_power='-1'
    fi
    # ipmitool dcmi power reading
    #echo "Printing results"
    if [ "${log_file}XXX" = "XXX" ]; then
        echo $proc_pid $date $stats $power $gpus_power
    else
        echo $proc_pid $date $stats $power $gpus_power >> $log_file
    fi
    sleep $interval
done

exit 0
