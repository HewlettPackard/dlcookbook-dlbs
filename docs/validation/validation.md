# __Validating benchmarks__

DLBS validates benchmarks before running them. Now, this is not 100% accurate. This means
even though experimenter may validate benchmarks successfully, some of them still may fail.
We are working on adding more rules to check benchmarks.

By default, validation is enabled. If experimenter does not validate benchmarks, it prints
out validation report and exits. Validation report will contain description of checks that
have failed. If you believe validation component is wrong, rerun experimenter with
`--no-validation` flag that apparently disables validation.

Currently, following checks are performed:

1. If containerized CPU based benchmarks are to be performed, validator ensures it can
   run `docker --version` command.
2. If containerized GPU based benchmarks are to be performed, validator ensures it can
   run `nvidia-docker --version` command.
3. If containerized benchmarks are to be performed, validator ensures docker image exists
   by running `docker inspect --type=image ${exp.docker_image}`
4. Validator checks there are no log file collisions i.e. there are no two or more
   benchmark experiments writing logs into same file.
5. For bare metal benchmarks (it sets up environment paths and library and python paths
   depending on experiment parameters):
      - For Tesorflow it ensures it can run the following command:
        ```bash
        python -c "import tensorflow as tf; print(tf.__version__;)"
        ```
      - For MXNet it ensures it can run the following command:
        ```bash
        python -c "import mxnet as mx; print(mx.__version__;)"
        ```
      - For Caffe2 it ensures it can run the following command:
        ```bash
        python -c "from caffe2.python.build import build_options; print(build_options);"
        ```
      - For Caffe's forks it ensures it can run the following commands:
        ```bash
        caffe --version
        ```
      - For TensorRT it ensures it can run the following commands:
        ```bash
        tensorrt --version
        ```


  We do not validate the following configurations now:
  1. Containerized frameworks.
  2. Availability of specified number of GPUs.

If validator validates benchmarks and some of them fails, log files should contain errors /
exceptions occurred during benchmark run. An indicator of a failed benchmark is the absence
of `results.time` key in a log file.

If you only want to validate configuration and not run it, use `validate` action
instead of `run`:
```bash
python ./python/dlbs/experimenter.py validate ...
```
