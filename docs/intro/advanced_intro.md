__Advanced introduction to Benchmarking Suite__
==============================================================================

Overview
--------

In this document we introduce [Deep Learning Benchmarking Suite
(DLBS)](https://github.com/HewlettPackard/dlcookbook-dlbs) and show step by step how one
can use this tool to perform end-to-end performance analysis of deep learning
workloads, from running benchmarks to reporting results.

Deep Learning Benchmarking Suite
--------------------------------

The [Deep Learning Benchmarking Suite](https://hewlettpackard.github.io/dlcookbook-dlbs) (DLBS) is a
collection of tools which assist in running DL benchmarks in a consistent and reproducible manner across
a range of software and hardware combinations. Out of the box the DLBS supports the following:

1.  Single-node, multi-GPU benchmarks.

2.  Seven DL frameworks including TensorFlow, BVLC/NVIDIA/Intel Caffe, Caffe2, MXNet and
    PyTorch and one inference engine NVIDIA TensorRT.

3.  Eighteen
    [models](https://hewlettpackard.github.io/dlcookbook-dlbs/\#/models/models?id=supported-models)
    for all supported frameworks. We try to make sure that a model
    implementation is consistent (the same) across all frameworks.

4.  Either bare metal or containerized frameworks including containers from
    [NVIDIA GPU Cloud.](https://ngc.nvidia.com/)

5.  [Basic
    dockerfiles](https://hewlettpackard.github.io/dlcookbook-dlbs/\#/docker/pull_build_images?id=buildpull-docker-images)
    for all frameworks which users can use to build containers with DL
    frameworks locally on their machines.

6.  Simple resource monitoring that tracks parameters such as CPU/GPU
    utilization, memory and power consumption etc.

7.  Basic reporting capabilities to build exploratory reports as well as reports
    that investigate both weak and strong scaling. Scripts to plot charts are
    also included. We plan to include advanced python notebook-based reporting
    capabilities in the nearest future.

More detailed information can be found on
[GitHub](https://github.com/HewlettPackard/dlcookbook-dlbs),
[documentation](https://hewlettpackard.github.io/dlcookbook-dlbs) and HPE
[developer](https://developer.hpe.com/platform/deep-learning-cookbook/home)
portals.

Installation
------------

1.  Install Docker and NVIDIA Docker for running containerized benchmarks. We
    have a quick overview
    [here](https://hewlettpackard.github.io/dlcookbook-dlbs/#/docker/docker?id=docker)
    on why we recommend using docker. If you want to use bare metal framework
    installations, skip all steps specific to containers.

2.  Clone Deep Learning Benchmarking Suite from
    [GitHub](https://github.com/HewlettPackard/dlcookbook-dlbs):

    ```bash
    git clone https://github.com/HewlettPackard/dlcookbook-dlbs dlbs
    ```

3.  The DLBS depends on modules from standard python library (python 2.7 only, python
    3.x is not supported currently). Optional dependencies that do not influence
    the benchmarking process are listed in `python/requirements.txt`. If these
    dependencies are not found, the code that uses it will be disabled.

4.  Build/pull docker images. If you do not have your own docker images, you can
    pull images or build images yourself. This
    [page](https://hewlettpackard.github.io/dlcookbook-dlbs/#/docker/pull_build_images?id=buildpull-docker-images)
    provides more details.

Benchmarking workflow
---------------------

1.  The user defines configuration which they want to explore.

    Configuration can be provided as a JSON file or command line arguments. In
    this post we will be using command line arguments for simplicity.
    Configuration can include definitions of frameworks, models, datasets etc.
    Once it is done,

2.  The user runs this configuration.

    Depending on exploration space, this process may take several minutes or
    several days. The result of this stage is a collection of raw textual log
    files. Those log files contain framework specific outputs as well as
    benchmark information logged by the DLBS. Then,

3.  The user runs the log parser.

    This is the part of DLBS that parses log files, extracts information and
    serializes it as a JSON file. There is an option to produce a compressed JSON
    file since textual JSON may be quite large.

4.  The user performs analysis of the results.

    The DLBS provides basic functionality for querying the JSON file and
    extracting the results of interest. The DLBS can also build basic
    exploratory, weak and strong scaling reports as well as plot various charts.

Before going through the benchmark steps described below, the user will need to
set up the benchmark directories. In the following description it will be
assumed that the user will be running benchmarks in the
*DLBS_ROOT/benchmarks/benchmark* folder where *DLBS_ROOT* is the root folder of
the DLBS it was cloned into:

```bash
# Go to the root folder of the DLBS
cd ./dlbs

# Create benchmark directories
mkdir -p ./benchmarks/benchmark

# Go to that folder
cd ./benchmarks/benchmark

# Setup host environment including python paths etc. Make sure you use Python 2.7.
source ../../scripts/environment.sh

python --version  # Make sure it is 2.7
echo $DLBS_NAME   # You should see here "Deep Learning Benchmarking Suite".
echo $DLBS_ROOT   # You should see here path to your root directory.

# Define shortcuts to scripts that we will use most
experimenter=../../python/dlbs/experimenter.py
parser=../../python/dlbs/logparser.py
```

Usually, for every series of benchmarks it is generally a good practice to
create a shell script that performs all these initializations so that they do not
need be reentered for each run. Also, note that the benchmarking directory can
reside at any place on the file system as long as the DLBS host environment is
properly initialized by calling the `environment.sh` script.

Getting help in command line
----------------------------

The DLBS provides basic help for input/output parameters and frameworks:

```bash
# Show help module functionality
python $experimenter help --help

# Show list of supported frameworks
python $experimenter help --frameworks

# Show list of commonly used parameters for tensorflow
python $experimenter help --frameworks tensorflow

# Show help message for parameter 'exp.gpus' (it can be a regular expression)
python $experimenter help --params exp.gpus

# Perform full text search in descriptions (it can be a regular expression)
python $experimenter help --text cuda
```

Do not worry if the output of the above-mentioned commands does not make sense
for now, the input and output parameters and their specifications will be
explained later in this post.

Benchmark configuration
-----------------------

In this particular example a series of benchmarks will be run on a 4 GPU
machine, using several frameworks and one neural network model. The command line
that launches benchmarks is as follows:

```bash
python $script run --log-level=info\
                   -Vexp.framework='["mxnet", "tensorflow", "caffe2"]'\
                   -Vexp.gpus='["0", "0,1", "0,1,2,3"]'\
                   -Vexp.model='["resnet50"]'\
                   -Vexp.replica_batch='[16]'\
                   -Pexp.docker=true\
                   -Pexp.num_warmup_batches=10\
                   -Pexp.num_batches=100\
                   -Pexp.phase='"training"'\
                   -Pexp.log_file='"${BENCH_ROOT}/logs/${exp.framework}/${exp.model}_${exp.num_gpus}_${exp.effective_batch}.log"'\
                   -Pmxnet.docker_image='"hpe/mxnet:cuda9-cudnn7"'\
                   -Pcaffe2.docker_image='"hpe/caffe2:cuda8-cudnn6"'\
                   -Ptensorflow.docker_image='"nvcr.io/nvidia/tensorflow:17.11"'
```

Configuration parameters are specified with `-V` and `-P` command line
arguments. The `V` parameters are called *variables*. The experimenter script
uses them to generate multiple benchmarks by computing the Cartesian product on
the range of values for variables. The `P` parameters are called parameters that
do not contribute to generating various benchmarks and just define values for
specific parameters. A particular configuration parameter can be a variable and
a parameter in different configurations depending on your needs.

The `V` and `P` arguments are followed by a parameter name and its value. Let's
consider in details configuration provided above. It defines 4 variables -
**exp.framework**, **exp.gpus**, **exp.model** and **exp.replica_batch**:

-   **exp.framework** A framework identifier to run.

-   **exp.gpus** A list of GPUs to use. If empty, CPU will be used instead.

-   **exp.model** A neural network model to benchmark.

-   **exp.replica_batch** A replica batch size. Another term for a device batch
    size.

> Variables values are usually assigned lists with different options. In the
example configuration shown above the variables are:

-   Three frameworks: *MXNet*, *TensorFlow* and *Caffe2*.

-   Three different sets of GPUs using 1, 2 and 4 GPUs respectively in
    particular combinations of GPU IDs (1 GPU: 0; 2 GPUs: 0, 1; 4 GPUs: 0, 1, 2,
    3.)

-   One value for the replica batch size, *16*; by default, experimenter uses
    weak scaling strategy.

-   Similarly only one value for the models *ResNet-50*.

So, given these configuration experimenter will run in total `3 * 3 * 1 * 1 = 9`
benchmarks.

The remaining configuration parameters are benchmark parameters that do not need
to be varied (i.e. they do not contribute to generating new benchmark
configurations) though they may have their specific values in different
benchmark like `exp.log_file` in the example above:

-   **exp.docker** A boolean parameter specifying if docker containers should be
    used.

-   **exp.num_warmup_batches** Number of warm-up batches to run.

-   **exp.num_batches** Number of benchmark batches to run.

-   **exp.phase** The benchmark phase (training/inference).

-   **exp.log_file** Benchmark log file. As it can be seen, parameters may refer
    other parameters. It’s similar to variable expansion in bash, though greatly
    simplified.

These are so called general parameters. There can be a framework specific
parameters that are used to simplify configurations. Framework specific
parameters belong to framework specific namespace (i.e. they start with
*framework.*):

-   **mxnet.docker_image** A docker image for *MXNet*.

-   **caffe2.docker_image** A docker image for *Caffe2*.

-   **tensorflow.docker_image** A docker image for *TensorFlow*.

>   The configuration used in this post is simplified and is not intended to be
>   used in real benchmarks. To use this configuration in a reasonable
>   benchmark, at a minimum, the number of warmup and benchmark batches need to
>   be increased.

>   DISCLAIMER: The benchmarks were ran on a machine with the GPUs' frequencies
>   reduced for maintenance reasons.

Parsing log files
-----------------

As specified by a `exp.log_file` parameter, log files produced by individual
benchmarks will be stored in *${BENCH_ROOT}/logs* directory. It may be
worthwhile spending some time browsing those files. In addition to framework
specific log output, e.g., the output from TensorFlow; they contain metadata
about the frameworks and experiments configuration parameters and variables;
system performance monitoring information, time series output of iteration
performance and system configuration information and parameters that were
specified by the user in the configuration command line or JSON configuration
files. Most importantly it will contain summarized information on the
performance and timing results of the benchmark in the *results.* namespace.

The log parser tool can parse those log files and print out information to a
console or JSON format file. This JSON file can then be used to plot charts and
build strong/weak/exploration reports. This JSON file will also be an input to
more advanced reporting tools that we plan to release as open source in the near
future.

To parse log files and extract all information, run the following command:

```bash
python $parser ./logs --recursive --output_file ./benchmarks.json
```

>   The log parser can write compressed files. To enable this, set file
>   extension to *json.gz*

Performing results analysis
---------------------------

### Weak scaling reports

The DLBS provides basic functionality for analyzing results. The following
command will generate a weak scaling report for benchmarks using the *MXNet*
framework with the *ResNet50* model:

```bash
reporter=../../python/dlbs/reports/summary_builder.py
python $reporter --summary-file ./benchmarks.json \
                 --type weak-scaling \
                 --target-variable results.time \
                 --query '{"exp.framework":"mxnet","exp.model":"resnet50"}'
```

It will print out several tables outlining the average batch times in
milliseconds, the throughput for various numbers of GPUs, and speedup and
efficiency estimates. If your benchmarks contain many more data, you can also
try to build a *strong-scaling* report. For single GPU training or inference
benchmark *exploration* this report will provide useful insights. The value for
target variable must be *results.time*. This is the output parameter that
contains a time in milliseconds for one batch. The *query* parameter specifies a
query that will select data points to build a report. It is a JSON dictionary
that maps keys (parameter names) to their values (constraints).

The following report will be printed:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Batch time (milliseconds)
Network             Batch     1          2          4
ResNet50            16        913.44     1029.24    1065.12

Inferences Per Second (IPS, throughput)
Network             Batch     1          2          4
ResNet50            16        17         31         60

Speedup (instances per second)
Network             Batch     1          2          4
ResNet50            16        1          1.82       3.53

Efficiency = 100% * t1 / tN
Network             Batch     1          2          4
ResNet50            16        100.00     88.74      85.75
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

### Building charts

To build graphical report, run the following command:

```bash
plotter=../../python/dlbs/reports/series_builder.py
python $plotter ./benchmarks.json \
       --xparam exp.num_gpus \
       --yparam results.throughput \
       --chart-file ./chart.png \
       --chart-type line \
       --series '[{"exp.framework":"mxnet","ecxp.model":"resnet50"},{"exp.framework":"caffe2","exp.model":"resnet50"},{"exp.framework":"tensorflow","exp.model":"resnet50"}]' \
       --aggregation avg \
       --chart-opts '{"title":"ResNet50 performance","xlabel":"Number of GPUs","ylabel":"Throughput","legend":["MXNet","Caffe2","TensorFlow"]}'
```

The following chart will be created:
<p align="center"><img src="./intro/imgs/chart.png"/></p>
Next Steps
----------

In this document we demonstrated how DLBS assists in running basic single-node
multi-GPU benchmarks. A number of tutorial scripts that can be used as examples
are located in a tutorials
[folder](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/tutorials/dlcookbook)
that demonstrate advanced usage of DLBS.
