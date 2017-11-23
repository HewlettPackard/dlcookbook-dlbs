export DLBS_NAME="Deep Learning Benchmarking Suite"
export DLBS_ROOT=$( cd $( dirname "${BASH_SOURCE[0]}" ) && cd .. && pwd )
export PATH=$DLBS_ROOT/scripts${PATH:+:}${PATH}
export PYTHONPATH=$DLBS_ROOT/python${PYTHONPATH:+:}${PYTHONPATH}
export experimenter="python $DLBS_ROOT/python/dlbs/experimenter.py"

[ ! -z $(which date) ] && export HAVE_DATE=true || export HAVE_DATE=false
[ ! -z $(which docker) ] && export HAVE_DOCKER=true || export HAVE_DOCKER=false
[ ! -z $(which nvidia-docker) ] && export HAVE_NVIDIA_DOCKER=true || export HAVE_NVIDIA_DOCKER=false
[ ! -z $(which python) ] && export HAVE_PYTHON=true || export HAVE_PYTHON=false
[ ! -z $(which awk) ] && export HAVE_AWK=true || export HAVE_AWK=false
[ ! -z $(which sed) ] && export HAVE_SED=true || export HAVE_SED=false

source $DLBS_ROOT/scripts/utils.sh
