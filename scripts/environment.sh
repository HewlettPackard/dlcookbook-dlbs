export DLBS_NAME="Deep Learning Benchmarking Suite"
export DLBS_ROOT=$( cd $( dirname "${BASH_SOURCE[0]}" ) && cd .. && pwd )
export PATH=$DLBS_ROOT/scripts${PATH:+:}${PATH}
export PYTHONPATH=$DLBS_ROOT/python${PYTHONPATH:+:}${PYTHONPATH}

# If not empty, will be used by CUDA runtime to store compiled kernels. In case of docker
# benchmarks, this folder is mounted inside docker container.
export CUDA_CACHE_PATH=/dev/shm/dlbs

# The main frontend benchmarking script. This is what's called DLBS.
export experimenter="$DLBS_ROOT/python/dlbs/experimenter.py"
#
export benchdata="$DLBS_ROOT/python/dlbs/bench_data.py"
# A script that parses log files.
export logparser="$DLBS_ROOT/python/dlbs/logparser.py"
# A script that builds textual perfformance reports.
export reporter="$DLBS_ROOT/python/dlbs/reports/summary_builder.py"
# This is a simple script that plots charts.
export plotter="$DLBS_ROOT/python/dlbs/reports/series_builder.py"

command -v date > /dev/null 2>&1 && export HAVE_DATE=true || export HAVE_DATE=false
command -v docker > /dev/null 2>&1 && export HAVE_DOCKER=true || export HAVE_DOCKER=false
command -v nvidia-docker > /dev/null 2>&1 && export HAVE_NVIDIA_DOCKER=true || export HAVE_NVIDIA_DOCKER=false
command -v python > /dev/null 2>&1 && export HAVE_PYTHON=true || export HAVE_PYTHON=false
command -v awk > /dev/null 2>&1 && export HAVE_AWK=true || export HAVE_AWK=false
command -v sed > /dev/null 2>&1 && export HAVE_SED=true || export HAVE_SED=false

source $DLBS_ROOT/scripts/utils.sh
