# __Tutorials__
## Running benchmarks
Simplest way to run benchmarks is to run `experimenter.py` script specifying benchmark parameters on a command line. Advanced usage may require specifying benchmark parameters in a json file.
All input parameters are logically divided into two types:

1. `Parameters` that do not need to be varied. They are specified on a command line with `-P` switch.
2. `Variables` that experimenter will vary. It's usually a list of values. They are specified on a command line with `-V` switch.

The following script runs one benchmark with BVLC Caffe framework:

```bash
python experimenter.py run \
    -Pexp.framework='"bvlc_caffe"'\
    -Pexp.env='"docker"'\
    -Pexp.phase='"training"'\
    -Pexp.gpus='0'\
    -Pexp.model='"alexnet"'\
    -Pexp.device_batch='"16"'\
    -Pexp.log_file='"./bvlc_caffe/training.log"'
```

The script will run one experiment with BVLC Caffe framework installed as a docker container. GPU 0 will be used. The BVLC Caffe will train the AlexNet model for some number of iterations with device batch size equal to 16 images. Results will be written to log file ./bvlc_caffe/training.log. Other parameters, such number of warmup iterations, number of benchmark iterations, concrete name of a docker image will have default values specified in the benchmark framework. No data will be used.

The following example demonstrates how variables are used:

```bash
python experimenter.py run \
    -Pexp.framework='"tensorflow"'\
    -Pexp.gpus='0'\
    -Vexp.env='["docker", "host"]'\
    -Pexp.log_file='"./tensorflow/${exp.env}/${exp.model}_${exp.effective_batch}.log"'\
    -Vexp.model='["alexnet", "googlenet"]'\
    -Vexp.device_batch='[2, 4]'\
    -Ptensorflow.docker.image='"hpe/tensorflow:centos-gpu-1.2.1"'
```

This script runs TensorFlow benchmarks with two models - AlexNet and GoogleNet. It runs benchmarks on both docker container and host. Different batch sizes are used - 2 and 4. This example also demonstrates that parameters may depend on other parameters. The experimenter will compute all variables and will terminate if some parameters are missing. It's similar to variable expansion mechanism in shell, though greatly simplified. In total, experimner will run `2 * 2 * 2 = 8` benchmarks. It will take all variables (`exp.env`, `exp.model`, `exp.device_batch`) and will compute Cartesian product. Then it will add other parameters to every generated experiment.

For additional examples, see the folder [tutorials/dlcookbook](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/tutorials/dlcookbook) that contains shell scripts with various examples.

## Parsing logs
For examples see [log_parser.sh](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/tutorials/dlcookbook/log_parser.sh) script.

## Building reports
For examples see the following scripts:

1. [summary_builder.sh](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/tutorials/dlcookbook/summary_builder.sh)
2. [time_analysis.sh](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/tutorials/dlcookbook/time_analysis.sh)
3. [bench_stats.sh](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/tutorials/dlcookbook/bench_stats.sh)
