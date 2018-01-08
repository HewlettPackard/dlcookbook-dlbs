# __Resource Monitor__

Deep Learning Benchmarking Suite provides basic (reference) functionality to
monitor resources while running benchmarks. There are several parameters that
enable monitoring:

1. `monitor.frequency` This parameter defines sampling frequency in seconds. Default
  value (0) disables resource monitor. Any positive number enables monitor. For instance,
  value of 0.1 sets sampling frequency to 100 ms.

2. `monitor.pid_folder` A folder that'll be used to communicate process identifiers
   between benchmarking and monitoring processes. The experimenter script will create a
   file in that folder named 'proc.pid' - if this file exists it will be overwritten
   and deleted on exit.

3. `monitor.timeseries` A configuration line that defines how experimenter will log
   data coming from resource monitoring script. Leave it to default value to collect all
   metrics we support.
   It's a string that specifies which timeseries metrics must go into a log file. Metrics
   are separated with comma (,). Each metric specification consists of three or four fields
   separated with colon (:) - `name:type:index_range`.
   - The `name` specifies timeseries name. The output parameter in a log file will have name  
     `results.use.$name`.
   - The `type` specifies how values that come from monitor need to be cast (std, int, float
     or bool).
   - Values from resource monitor come as a whitespace separated string. The index range
     specifies how that maps to a timeseries name. It can be a single integer(for instance
     `time:str:1`) specifying exact index or a index and number of elements that should be
     appended to a timeseries item. Number of elements may not be present what means scan
     until the end of list is reached (for instance `gpu:float:8:2` or `gpu:float:8:`). If
     number of elements is specified, a timeseries will contain items that will be lists
     even though number of elements may be 1."

### Technical details
The resource monitoring script is located [here](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/scripts/resource_monitor.sh).
Resource monitor is a bash script that tracks system resources based on process id (pid).
If resource monitor is enabled, the backend writes to a particular file a pid of a benchmarking
process `${monitor.pid_folder}/proc.pid`. Monitor with specified frequency reads this file. If it's
empty, monitor sleeps for short period of time and reads again. If file contains a token
`exit`, monitor exists. Else, if file contains a single integer number, monitor considers this to be a process id.
At every sampling moment, it outputs a white space separated string with the following metrics:
```
process_pid time_stamp virt res shrd cpu mem power gpu0_power gpu1_power ... gpuN_power
```
where:

1. `process_pid` is the process PID of parent benchmarking process. The actual set of
   processes `proc_pids` to monitor is determined based in parent process id `proc_pid`:
   ```bash
    sproc_pids=$(pgrep -P ${proc_pid} -d,)
    if [ "$sproc_pids" == "" ]; then
	       proc_pids=$proc_pid
    else
	       proc_pids=$proc_pid,$sproc_pids
    fi
   ```
2. `time_stamp` is the time stamp in the format `+%Y-%m-%d:%H:%M:%S:%3N`:
   ```bash
   date +%Y-%m-%d:%H:%M:%S:%3N
   ```
3. `virt` is the virtual image: the  total  amount  of  virtual  memory  used  by the task.  
   It includes all code, data and shared libraries  plus  pages  that have  been  swapped
   out and pages that have been mapped but not used.
4. `res` is the resident size: the non-swapped physical memory a task has used.
5. `shrd` is the shared mem size: the amount of shared memory used by a task.  It simply
   reflects memory that could be potentially shared with other processes.
6. `cpu` is the CPU usage: total  CPU  time  the  task  has  used  since it started.  When
    'Cumulative mode' is On, each process is listed  with  the  cpu time  that  it  and  its
    dead  children  has used.  You toggle 'Cumulative mode' with 'S', which is a command-lin
    option  and an  interactive  command.   See the 'S' interactive command for
    additional information regarding this mode.
7. `mem` is the memory usage: a task's currently used share of available physical memory.
    Memory and CPU consumption is determine like this:
    ```bash
    stats=$(top -b -p${proc_pids} -n1 | tail -n +8)
    stats=$(echo "$stats" | awk '{virt += $5; res += $6; shrd += $7; cpu += $9; mem += $10}\
          END {print virt, res, shrd, cpu, mem}')
    ```
8. `power` current power consumption (excluding GPUs):
   ```bash
   ipmitool dcmi power reading | grep "Instantaneous" | awk '{print $4}'
   ```
9. `gpuN_power` depending on number of GPUs in a system, a per-GPU power consumption:
   ```bash
   nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits
   ```

### Output
Given that resource monitor is enabled, the following parameters will be in log files:
```python
results.use.mem_virt=[49.48, 49.719, 49.859, ...]
results.use.power=[-1.0, -1.0, -1.0, ...]
results.use.mem=[0.3, 1.1, 1.7, ...]
results.use.mem_shrd_=[95796.0, 164500.0, 196124.0, ...]
results.use.mem_res=[111788.0, 365640.0, 542832.0, ...]
results.use.time=["2017-12-14:14:00:41:729", "2017-12-14:14:00:42:008", ...]
results.use.gpus=[[17.7], [69.56], [78.12], ...]
results.use.cpu_=[586.7, 100.0, 100.0, ...]
```

### Limitations
In current implementation, the resource monitor is enabled if it is enabled for a
first benchmark in a list of benchmarks and remains enabled for all other benchmarks.
So, it's not possible now to selectively enable/disable resource monitor on a per-benchmark
basis - it's either enabled or disabled for all benchmarks. This however is very easy to fix.

### Replacing monitor with advanced version
It should be possible to replace standard resource monitor with a custom one assuming
it is implemented in a similar way and outputs a whitespace separated list of metrics.
A parameter `monitor.launcher` defines a full path to bash script/program that implements
resource monitor. Also, a `monitor.timeseries` parameter should be adjusted accordingly.
This program must accept three parameters:

1. `pid_file` - a textual file name that may be in three states:
   1. `empty` - no benchmarking process is being run
   2. `pid` - a single integer number defining parent benchmarking process
   3. `exit` - benchmarking process done, must exit
2. `log_file` - a log file to output metrics, if empty script must print to a standard output.
3. `interval_seconds` - sampling frequency in seconds.

For instance, a reference implementation may be called like this:
```bash
./scripts/resource_monitor.sh /dev/shm/monitor/proc.pid "" 0.1
```
