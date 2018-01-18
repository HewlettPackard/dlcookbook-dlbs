# __System Information__

Benchmarking suite provides basic functionality to record system configuration.
A parameter `exp.sys_info` defines what should be collected. Default value - empty
string - disables this functionality.

The value of this parameter is a comma separated string that defines what tools
should be used to collect system wide information. To collect all information use:

```bash
-Pexp.sys_info='"inxi,cpuinfo,meminfo,lscpu,nvidiasmi"'
```
The following source of information are supported:

1. `inxi` The _inxi_ tool must be available (https://github.com/smxi/inxi). We record
an output of `inxi -Fbfrlp`.
2. `cpuinfo` The content of _/proc/cpuinfo_
3. `meminfo` The content of _/proc/meminfo_
4. `lscpu` The output of `lscpu`
5. `nvidiasmi` The output of `/usr/bin/nvidia-smi -q`

All information is stored in a `hw` namespace i.e. `hw.inxi`, `hw.cpuinfo`, `hw.meminfo`,
`hw.lscpu` and `hw.nvidiasmi` as json objects.

In addition, a complete output in a json format can be obtained with:

```bash
python ./python/dlbs/experimenter.py sysinfo
```

The class responsible for logging this information is [SysInfo](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/python/dlbs/sysinfo/systemconfig.py).
