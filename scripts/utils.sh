#-------------------------------------------------------------------------------
# To return TRUE value, return 0 (success), for FALSE return 1 (failure)
#-------------------------------------------------------------------------------

assert_docker_img_exists() {
  [ "$#" -ne 1 ] && logfatal "assert_docker_img_exists: one argument expected";
  [ -z "$(docker images -q $1)" ] && \
   logfatal "docker image \"$1\" does not exist locally, \
             pull it from hub or build it manually" || return 0;
}
assert_not_docker_and_singularity() {
  [ "$exp_docker"  = true -a "$exp_singularity" = true ] && \
   logfatal "Both exp.docker and exp.singularity were set to true, however, only one container type can be selected." || return 0;
}

assert_singularity_img_exists() {
  [ "$#" -ne 1 ] && logfatal "assert_singularity_img_exists: one argument expected";
  [ ! -f $1 ] && \
   echo "singularity image \"$1\" does not exist locally, pull it from hub or build it manually" || return 0;
}

export -f assert_not_docker_and_singularity
export -f assert_docker_img_exists
export -f assert_singularity_img_exists

# This can be used like: loginfo "..."
timestamp() { date +'%m-%d-%y %H:%M:%S';}
loginfo() { echo "$(timestamp) [INFO]    $1";}
logwarn() { echo "$(timestamp) [WARNING] $1";}
logerr()  { echo "$(timestamp) [ERROR]   $1";}
logfatal(){ echo "$(timestamp) [FATAL]   $1"; exit 1; }
export -f timestamp loginfo logwarn logerr logfatal

logvars() {
  local arg var vars name value
  for arg in "$@"; do
    vars=("$arg")
    for var in $vars; do
      name=$var
      value=${!var}
      loginfo "$name: \"$value\""
    done
  done
}
echo_vars() {
  local arg var vars name value
  for arg in "$@"; do
    vars=("$arg")
    for var in $vars; do
      name=$var
      value=${!var}
      echo "$name: \"$value\""
    done
  done
}
export -f logvars echo_vars

dir_empty() {
  [ "$#" -ne 1 ] && logfatal "dir_empty: one argument expected";
  [ "$(ls -A $1)" ] && return 1 || return 0;
}
dir_exists() {
  [ "$#" -ne 1 ] && logfatal "dir_exists: one argument expected";
  [ -d "$1" ] && return 0 || return 1;
}
file_exists() {
  [ "$#" -ne 1 ] && logfatal "file_exists: one argument expected";
  [ -f "$1" ] && return 0 || return 1;
}
export -f dir_empty dir_exists file_exists

remove_files() {
  local file
  for file in "$@"; do
    if [ -f "$file" ]; then
      rm -f $file || logfatal "remove_files: file \"$1\" was not removed"
    fi
  done
  return 0
}
export -f remove_files
create_dirs() {
  local dir
  for dir in "$@"; do
    if [ ! -d "$dir" ]; then
      mkdir -p $dir || logfatal "create_dirs: directory \"$1\" was not created"
    fi
  done
  return 0
}
export -f create_dirs
assert_files_exist() {
  local file
  for file in "$@"; do
    [ -f "$file" ]  || logfatal "assert_files_exist failed on \"$file\"";
  done
  return 0
}
export -f assert_files_exist
assert_dirs_exist() {
  local dir
  for dir in "$@"; do
    [ -d "$dir" ] || logfatal "assert_dirs_exist failed on \"$dir\"";
  done
  return 0
}
export -f assert_dirs_exist
assert_funcs_exist() {
  local func
  for func in "$@"; do
    declare -f $func >> /dev/null || logfatal "assert_funcs_exist: $func not defined";
  done
  return 0
}
export -f assert_funcs_exist


# Extract a value given its key from a file. Key-Value format is:
# key: "value"
# Returns first occurence.
get_value_by_key() {
  [ "$#" -ne 2 ] && logfatal "get_value_by_key: 2 arguments expected";
  sed -nr "s/^$2:[ ]+\"([^\"]+)\"/\1/p" < $1 | head -1
}
export -f get_value_by_key

tf_version() {
  [ "$#" -ne 1 ] && logfatal "tf_version: one argument expected";
  nvidia-docker run -i $1 python -c 'import tensorflow as tf; print(tf.__version__);' && return 0 || return 1;
}
tf_devices() {
  [ "$#" -ne 1 ] && logfatal "tf_devices: one argument expected";
  nvidia-docker run -i $1 python -c 'from tensorflow.python.client import device_lib; print([x.name for x in device_lib.list_local_devices()]);' 2>/dev/null && return 0 || return 1;
}
export -f tf_version tf_devices

caffe2_error() {
  [ "$#" -ne 1 ] && logfatal "caffe_error: one argument expected";
  # -q, --quiet, --silent
  # Quiet; do not write anything to standard output. Exit immediately with zero
  # status if any match is found, even if an error was detected.
  [ ! -f "$1" ] && return 0;
  #grep -q "^RuntimeError:" "$1" && return 0 || return 1;
  grep -q "^__results.time__" "$1" && return 1 || return 0;
}
tf_error() {
    [ "$#" -ne 1 ] && logfatal "tf_error: one argument expected";
    [ ! -f "$1" ] && return 0;
    # -q, --quiet, --silent
    # Quiet; do not write anything to standard output. Exit immediately with zero
    # status if any match is found, even if an error was detected.
    grep -q "ResourceExhaustedError\|core dumped\|std::bad_alloc" "$1" && return 0 || return 1;
}
caffe_error() {
  [ "$#" -ne 1 ] && logfatal "caffe_error: one argument expected";
  # -q, --quiet, --silent
  # Quiet; do not write anything to standard output. Exit immediately with zero
  # status if any match is found, even if an error was detected.
  [ ! -f "$1" ] && return 0;
  grep -q "*** Check failure stack trace: ***" "$1" && return 0 || return 1;
}
mxnet_error() {
  [ "$#" -ne 2 ] && logfatal "mxnet_error: two arguments expected (log file and phase)";
  # -q, --quiet, --silent
  # Quiet; do not write anything to standard output. Exit immediately with zero
  # status if any match is found, even if an error was detected.
  [ ! -f "$1" ] && return 0;
  grep -q "^__results.time__" "$1" && return 1 || return 0;
}
export -f caffe2_error tf_error caffe_error mxnet_error

# $1: exp_dir, $2 batch size
is_batch_good() {
  [ "$#" -ne 2 ] && logfatal "is_batch_good: two arguments expected (file name and device batch)";
  if [ ! -f "$1" ]; then
    return 0;       # File does not exist, no error information
  else
    local current_batch=$(cat $1)
    if [ "$2" -lt "$current_batch" ]; then
      return 0      # File exists, error batch is larger
    fi
    return 1        # File exists, error batch is smaller
  fi
}
export -f is_batch_good
# $1: exp_dir, $2 batch size
update_error_file() {
  [ "$#" -ne 2 ] && logfatal "update_error_file: two arguments expected (file name and device batch)";
  if [ ! -f "$1" ]; then
    echo "$2" > $1
  else
    local current_batch=$(cat $1)
    if [ "$2" -lt "$current_batch" ]; then
      echo "$2" > $1
    fi
  fi
}
export -f update_error_file

# The 'exp_file' may be opened during this call.
caffe_postprocess_log() {
  [ "$#" -ne 6 ] && logfatal "caffe_postprocess_log: 6 arguments expected";
  local exp_file=$1  batch_file=$2 phase=$3 device_batch=$4 effective_batch=$5 iters=$6

  [ ! -f "$exp_file" ] && return 0;
  # Check file exists
  # Delete from file ^M symbols (carriage return)
  # We can't really do this for now. File is opened. - Well, actually we can. In new
  # version file is not opened here.
  #sed -i 's/\r//g' $exp_file
  if caffe_error "$exp_file"; then
    logwarn "error in \"$exp_file\" with effective batch $effective_batch (per device batch $device_batch)";
    update_error_file "$batch_file" "$device_batch";

    local error_hint=$(grep -wo "Check failed: .*" $exp_file | head -1)
    echo "__exp.status__= \"failure\"" >> $exp_file
    echo "__exp.status_msg__= \"Error has been found in Caffe log file ($error_hint).\"" >> $exp_file
    return 1;
  else
    # Here we know if we are in time or train mode.
    if [ "$phase" = "inference" ]; then
      local tm=$(grep "Average Forward pass:" $exp_file | awk '{print $8}')
      #local btm=$(grep "Average Backward pass:" $exp_file | awk '{print $8}')
      #local fbtm=$(grep "Average Forward-Backward:" $exp_file | awk '{print $7}')
      #echo "__results.time__= $ftm" >> $exp_file
      #echo "__results.backward_time__= $btm" >> $exp_file
      #echo "__results.training_time_approx__= $fbtm" >> $exp_file
    else
      # hours:minutes:seconds:milliseconds
      #stm=$(grep "Starting Optimization" $exp_file | awk '{print $2}' | head -1 | xargs date -u +"%s.%N" -d)
      local stm=$(grep "Solving " $exp_file | awk '{print $2}' | tail -1 | xargs date -u +"%s.%N" -d)
      local etm=$(grep "Optimization Done" $exp_file | awk '{print $2}' | tail -1 | xargs date -u +"%s.%N" -d)
      local tm=$(echo $stm $etm $iters | awk '{printf "%f", 1000.0*($2-$1)/$3}')
      #echo "__results.time__= $fbtm" >> $exp_file
    fi
    local throughput=$(echo $tm $effective_batch | awk '{printf "%f", 1000.0 * $2 / $1}')
    echo "__results.time__= $tm" >> $exp_file
    echo "__results.throughput__= $throughput" >> $exp_file
  fi
  return 0
}
export -f caffe_postprocess_log

# report_and_exit  status  status_message  log_file
report_and_exit() {
  [ "$#" -ne 3 ] && logfatal "report_and_exit: 3 arguments expected (status, status message and log file)";
  echo "__exp.status__= \"$1\"" >> $3
  echo "__exp.status_msg__= \"$2\"" >> $3
  logfatal "$2 (status code = $1)"
}
export -f report_and_exit
