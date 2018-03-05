# __Tutorials__
## Running benchmarks
Simplest way to run benchmarks is to run `experimenter.py` script specifying benchmark parameters on a command line. Advanced usage may require specifying benchmark parameters in a json file.
All input parameters are logically divided into two types:

1. `Parameters` that do not need to be varied. They are specified on a command line with `-P` switch.
2. `Variables` that experimenter will vary. It's usually a list of values. They are specified on a command line with `-V` switch.

The following script runs one benchmark with BVLC Caffe framework:

```bash
python ./python/dlbs/experimenter.py run \
             -Pexp.framework='"bvlc_caffe"'\
             -Pexp.docker=true\
             -Pexp.phase='"training"'\
             -Pexp.gpus='"0"'\
             -Pexp.model='"alexnet"'\
             -Pexp.replica_batch='"16"'\
             -Pexp.log_file='"./benchmarks/my_experiment/bvlc_caffe.log"'
```

The script will run one experiment with BVLC Caffe framework (`-Pexp.framework='"bvlc_caffe"'`) installed as a docker container (`-Pexp.docker=true`). GPU 0 will be used (`-Pexp.gpus='"0"'`). The BVLC Caffe will train the AlexNet model (`-Pexp.model='"alexnet"'`) for some number of iterations with per replica batch size equal to 16 images (`-Pexp.replica_batch='"16"'`). Results will be written to log file (`-Pexp.log_file='"./benchmarks/my_experiment/bvlc_caffe.log"'`). Other parameters, such as number of warmup iterations, number of benchmark iterations, concrete name of a docker image will have default values specified in the benchmark framework. No real data will be used.

The following example demonstrates how variables are used:

```bash
python ./python/dlbs/experimenter.py run\
             -Pexp.framework='"tensorflow"'\
             -Pexp.gpus='"0"'\
             -Vexp.docker=true\
             -Pexp.log_file='"./benchmarks/my_experiment/tf_${exp.model}_${exp.replica_batch}.log"'\
             -Vexp.model='["alexnet", "googlenet"]'\
             -Vexp.replica_batch='[2, 4]'\
             -Ptensorflow.docker_image='"hpe/tensorflow:cuda9-cudnn7"'
```
This script runs TensorFlow benchmarks with two models - AlexNet and GoogleNet. Different batch sizes are used - `2` and `4`. This example also demonstrates that parameters may depend on other parameters. The experimenter will compute all variables and will terminate if some parameters are missing. It's similar to variable expansion mechanism in shell, though greatly simplified. In total, experimenter will run `2 * 2 = 4` benchmarks. It will take all variables (`exp.model` and `exp.replica_batch`) and will compute Cartesian product. Then it will add other parameters to every generated experiment.

For additional examples, see the folder [tutorials/dlcookbook](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/tutorials/dlcookbook) that contains shell scripts with various examples.

## Parsing logs
For examples see [log_parser.sh](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/tutorials/dlcookbook/log_parser.sh) script.

## Building reports
For examples see the following scripts:

1. [summary_builder.sh](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/tutorials/dlcookbook/summary_builder.sh)
2. [time_analysis.sh](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/tutorials/dlcookbook/time_analysis.sh)
3. [bench_stats.sh](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/tutorials/dlcookbook/bench_stats.sh)
