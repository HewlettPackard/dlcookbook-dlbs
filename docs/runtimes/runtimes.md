# __Runtimes__

DLBS can run bare metal benchmarks or inside docker containers. The boolean `exp.docker` parameter defines how DLBS runs benchmarks. By default, this value is `true` and if you need to use docker, there's no need to provide this parameter.

## Bare metal benchmarks
To run bare metal benchmarks, set the value of the `exp.docker` parameter to `false` either on a command line `-Pexp.docker=false` or in a JSON configuration file `"exp.docker": false`.

## Docker benchmarks
To run benchmarks inside docker containers, set the value of the `exp.docker` parameter to `true`: `-Pexp.docker=true` in case of a command line or `"exp.docker": true` if a JSON configuration is used (default  value of this parameter is `true`).

With docker, we generally have three options to run it - with `docker`, `nvidia-docker` or `nvidia-docker2`.

##### CPU benchmarks
All CPU based benchmark by default use `docker`. This is defined by a value of the `exp.device_type` parameter:
```json
"exp.docker_launcher": {
      "val": "$('nvidia-docker' if '${exp.device_type}' == 'gpu' else 'docker')$",
}
```

##### GPU benchmarks
As it can be seen from above example, all GPU based benchmarks by default use `nvidia-docker`. This introduces an issue in environments where nvidia-docker2 is installed and nvidia-docker shell script is missing (not available).

In these cases, where nvidia-docker is unavailable or does not work, manual configuration needs to be provided (assuming exp.docker is true):
```json
    "exp.docker_launcher": "docker",
    "exp.docker_args": "--rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --ipc=host --runtime=nvidia"
```
The same values can be provided on a command line with `-P` switch. So, basically, two things happen here:
1. Docker executable is set to `docker`;
2. A docker parameter `runtime` is set to `nvidia`: `--runtime=nvidia`.

The `exp.docker_args` parameter has some default [value](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/dlbs/configs/base.json#L282), so, it need to be replicated here. If this is not done, default parameters will be lost.
