**System Information**
======================

Overview
--------

DLBS provides basic functionality to store system configuration in a benchmark log file. A parameter `exp.sys_info` defines what should be collected. Default value (empty string) disables this functionality.

The value of this parameter is a comma separated string that defines what modules should be used to collect system information. To collect all information use:

```
-Pexp.sys_info='"inxi,cpuinfo,meminfo,lscpu,nvidiasmi,dmi"'
```

The following modules are supported:

1.  `inxi` The output of *inxi* tool that must be available (https://github.com/smxi/inxi). We record an output of `inxi -Fbfrlp`.

2.  `cpuinfo` The content of */proc/cpuinfo*

3.  `meminfo` The content of */proc/meminfo*

4.  `lscpu` The output of `lscpu`

5.  `nvidiasmi` The output of `/usr/bin/nvidia-smi -q`

6.  `dmi` Stores contents of various files in */sys/devices/virtual/dmi/id* that does not require *sudo* access. In current implementation, it is the following files: *board_name*, *board_vendor*, *product_name* and *sys_vendor*.

All information is stored in a `hw` namespace i.e. `hw.inxi`, `hw.cpuinfo`, `hw.meminfo`, `hw.lscpu`,`hw.nvidiasmi` and `hw.dmi` as json objects. The information is stored in pretty much unstructured form. We do simple post processing of results though. For now, the primary goal is to store this information and work out best practices on how to use it and how to store it in a more structured way.

In addition, a complete output in a json format can be obtained with:

```
python ./python/dlbs/experimenter.py sysinfo
```

The class responsible for logging this information is [SysInfo](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/python/dlbs/sysinfo/systemconfig.py).

Example
-------

The example output of `dmi` module:

```json
"hw.dmi": {
    "board_name": "158B",
    "sys_vendor": "Hewlett-Packard",
    "board_vendor": "Hewlett-Packard",
    "product_name": "HP Z820 Workstation"
}
```

This information will be parsed by the log parser and will be stored as a dictionary under `hw.dmi` key.
