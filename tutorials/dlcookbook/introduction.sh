#!/bin/bash

# Set the root benchmarking directory. This variable may be used in json configuration
# files to build different relative paths. It should not necceserily be this
# directory where this script is located. You can define any other variables here
# and use them in json configuration files like ${VAR_NAME}.
export BENCH_ROOT=$( cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )

# This is a very useful variable. Can significantly speed-up the initialization
# time.
export CUDA_CACHE_PATH=""

# We need to run this to setup paths and environment. In most cases, you want to
# source common.sh in scripts/ folder of this project. This script will initialize
# DLBS_ROOT variable that will point to the root folder of the dlcookbook project.
. ${BENCH_ROOT}/../../scripts/environment.sh

script=$DLBS_ROOT/python/dlbs/experimenter.py
#------------------------------------------------------------------------------#
# Example: printing help.
if true; then
    python $script --help
fi

#------------------------------------------------------------------------------#
# Example: load default configuration, pretty print it to a command line and exit.
# Without other arguments, it will print default configuration. Parameters and
# variables defined in configuration files will not be evaluated. The 'print-config'
# just prints what's inside configuration files i.e. parameters/variables passed
# via comamnd line arguments will not be included (to do this, a plan must be built,
# see below).
if false; then
    python $script print-config --log-level=debug
fi

#------------------------------------------------------------------------------#
# Example: it's possible to discard the default configuration, what probably does
# not make sense in majority of scenarios. Should print empty object
if false; then
    python $script print-config --discard-default-config
fi

#------------------------------------------------------------------------------#
# Example: to run experiments, we need to build its plan. Plan basically describes
# multiple experiments derived from provided configuration. To build plan, we
# need to use action "build". If no plan file is specified (--plan), generated
# plan will be printed out to a console. With discard option, we should get empty plan
if false; then
    python $script build --discard-default-config
fi

#------------------------------------------------------------------------------#
# Example: There are two types of variables. The first type is 'parameter'
# variable or just parameter. These parameters do not contribute to generating
# different experiments and may be common to all experiments. It's possible to
# specify them on a command line. All values of such paarmeters must be json
# parsable (json.loads()).
if false; then
    python $script build --discard-default-config --log-level=debug \
                         -Pstr.greeting='"Hello World!"' -Pint.value=3\
                         -Pfloat.value=3.4343 -Plist.value='["1", "2", "3"]'\
                         -Plist.value2='[100,101,102]'
fi

#------------------------------------------------------------------------------#
# Example: other type of variables is just variables. They contribute to generating
# experiments variations. Pay attention to the number of generated experiments. It
# must be 8 ( = 2 * 4).
if false; then
    python $script build --discard-default-config --log-level=debug\
                         -Vexp.framework='["tensorflow", "caffe2"]'\
                         -Vexp.replica_batch='[1, 2, 4, 8]'
fi

#------------------------------------------------------------------------------#
# Example: and of course, we can combine parameters and variables.
if false; then
    python $script build --discard-default-config --log-level=debug\
                         -Pexp.num_batches=1000\
                         -Vexp.framework='["tensorflow", "caffe2"]'\
                         -Vexp.replica_batch='[1, 2, 4, 8]'
fi

#------------------------------------------------------------------------------#
# Example: there's a "dummy" script that just prints all its command line
# arguments. This is a simple example how we use configuration parameters to
# conduct experiments:
if false; then
    action=run # Valid valus: 'build', 'run'
    python $script $action --discard-default-config --log-level=debug\
                           -Vexp.framework='"dummy"'\
                           -Vexp.greeting='["Hello!", "How are you?"]'\
                           -Pdummy.launcher='"${DLBS_ROOT}/scripts/launchers/dummy.sh"'
fi

#------------------------------------------------------------------------------#
# Example: extensions can be specified on a command line
if false; then
    python $script build   --discard-default-config --log-level=debug\
                           -Vexp.framework='"dummy"'\
                           -Vexp.greeting='["Hello!", "How are you?"]'\
                           -Pdummy.launcher='"${DLBS_ROOT}/scripts/launchers/dummy.sh"'\
                           -E'{"condition":{"exp.greeting":"Hello!"}, "parameters": {"exp.greeting.extention": "You should see me only when exp.greeting is Hello!"}}'
fi
