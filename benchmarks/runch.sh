#!/bin/bash

do_run=false
do_parse=false
projID=""
configfile=""
script_dir=""
action=""
keep=false
framework=""
freq=""

usage() {
        echo "Usage:"
        echo "      ${BASH_SOURCE[0]} [-a action | -f framework -c configfile -p projID | -s /path/to/scripts | h ]"
        echo "      -a run | validate | build will execute experimenter with action"
        echo "      -c config_file - JSON configuration file."
        echo "      -f framework"
        echo "      -F frequency"
        echo "      -l will execute logparser"
        echo "      -k do not delete the results directory if it already exists"
        echo "      -p projID - projectID"
        echo "      -s /path/to/scripts - specify the path to the dlbs/scripts directory to source environment.sh"
        echo
}

while getopts ":a:f:F:lkc:p:s:h" opt; do
  case $opt in
    a)
      action=$OPTARG
      ;;
    k)
      keep=true
      ;;
    f)
      framework=$OPTARG
      ;;
    F)
      freq=$OPTARG
      ;;
    c)
      configfile=$OPTARG
      ;;

    p)
      projID=$OPTARG
      ;;
    l)
      do_parse=true
      ;;
    s)
      SCRIPT_DIR=$OPTARG
      ;;
    h)
        usage
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      usage
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      usage
      exit 1
      ;;
  esac
done
if  [ ! -z "$action" ] ; then
    if  [ -z "$configfile" ] ; then
        echo "Must specify a configuration file."
        usage
        exit -1
    fi
    echo "Will execute experimenter $action."
fi

if  [ -z "$freq" ] ; then
    echo "Must specify a GPU application clock frequency."
    usage
    exit -1
fi

if  [ -z "$framework" ] ; then
    echo "Must specify a framework."
    usage
    exit -1
fi
if  [ "$do_parse" = true ] ; then
    echo "Will run the log parser".
fi

export CUDA_CACHE_PATH=/dev/shm/cuda_cache
#------------------------------------------------------------------------------#
export BENCH_ROOT=$( cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )
if [ -z "$SCRIPT_DIR" ];then
    echo from bench_root
    source ${BENCH_ROOT}/../scripts/environment.sh
else
    source ${SCRIPT_DIR}/environment.sh
fi

script=$DLBS_ROOT/python/dlbs/experimenter.py
parser=$DLBS_ROOT/python/dlbs/logparser.py

loglevel=warning
#------------------------------------------------------------------------------#
# For more detailed comments see './bvlc_caffe.sh' script.
#------------------------------------------------------------------------------#
# Example: a minimal working example to run TensorFlow. Run one experiment and
# store results in a file.
# If you run multiple experiments, you really want to make sure that experiment
# log file is different for every experiment.
if  [ ! -z "$action" ] ; then
    if  [ "$action" = "run" ] ; then
		#nvidia-smi --applications-clocks=715,${freq}
        if  [ "$keep" = false ] ; then
            rm -rf $framework
        fi
    fi
    python $script $action --log-level=$loglevel --config ${configfile} -Pexp.gpufreq='"'${freq}'"' -Pexp.proj='"'$projID'"'
fi
if  [ "$do_parse" = true ] ; then
    python $parser --recursive --log-dir ./$framework
fi
