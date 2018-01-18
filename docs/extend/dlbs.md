# __Extending Deep Learning Benchmarking Suite__

## New framework

By default, experimenter supports variety of frameworks. If a particular framework, probably, a custom version of a supported framework, is not supported, it can be added as a new backend to experimenter. There are several steps to add support for a new framework:

1. Write benchmarking framework similar to [mxnet](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/python/mxnet_benchmarks) or [caffe2](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/python/caffe2_benchmarks) benchmarks projects. The goal is to have a script, that can run experiments/train models given some input command line parameters. Those parameters may include model name, device batch size, data directory, gradient aggregation schema etc.
2. Write a JSON configuration file that will define parameters and that will build parameters that depend on other parameters.
3. Write a wrapper shell script that will accept command line parameters from experimenter and that will call framework script translating experimenter parameters to framework parameters.

Every framework has its own unique name like `tensorflow`, `caffe2`, `bvlc_caffe` etc. A parameter `exp.framework` defines framework to use. Experimenter assumes that all framework specific parameters are defined in the `${exp.framework}` namespace.

For simplicity, let's assume we want to add support for Caffe framework. It turns out that we actually can use caffe directly by invoking caffe with specific parameters to run experiments. So, we can skip first step and start with the second one.
This is the default configuration of Caffe that is included into experimenter:
```json
{
  "parameters": {
    "caffe.launcher": {
      "val":  "${DLBS_ROOT}/scripts/launchers/caffe.sh",
      "type": "str",
      "desc": "Path to script that launches Caffe benchmarks."
    },
    "caffe.env": {
      "val":  "${runtime.EXPORT_CUDA_CACHE_PATH}",
      "type": "str",
      "desc": "An string that defines export variables that's used to initialize environment for Caffe."
    },
    "caffe.fork": {
      "val":  "bvlc",
      "type": "str",
      "desc": [
        "A fork of Caffe. Possible values may vary including 'bvlc', 'nvidia' and 'intel'. The value",
        "of this parameter is computed automatically. The only reason why it has default value (bvlc)",
        "is because there are other parameters referencing this parameter that need to be resolved."
      ]
    },
    "caffe.action": {
      "val":  "$('train' if '${exp.phase}' == 'training' else 'time')$",
      "type": "str",
      "val_domain": ["train", "time"],
      "desc": "Action that needs to be performed by caffe. Possible values are 'train' or 'time'."
    },
    "caffe.model_file": {
      "val":  "${exp.id}.model.prototxt",
      "type": "str",
      "desc": "Caffe's prototxt model file."
    },
    "caffe.solver_file": {
      "val":  "${exp.id}.solver.prototxt",
      "type": "str",
      "desc": "Caffe's prototxt solver file."
    },
    "caffe.model_dir": {
      "val":  "$('${DLBS_ROOT}/models/${exp.model}' if not ${exp.docker} else '/workspace/model')$",
      "type": "str",
      "desc": "Directory where Caffe's model file is located. Different for host/docker benchmarks."
    },
    "caffe.solver": {
      "val": [
        "net: '${caffe.model_dir}/${caffe.model_file}'\\n",
        "max_iter: ${exp.num_batches}\\n",
        "test_interval: 0\\n",
        "snapshot: 0\\n",
        "snapshot_after_train: false\\n",
        "base_lr: 0.01\\n",
        "lr_policy: 'fixed'\\n",
        "solver_mode: $('${exp.device_type}'.upper())$\\n",
        "$('solver_data_type: ${nvidia_caffe.solver_precision}' if '${exp.framework}' == 'nvidia_caffe' else '')$"
      ],
      "type": "str",
      "desc": "A content for a Caffe's solver file in case Caffe benchmarks train phase."
    },
    "caffe.args": {
      "val": [
        "$('--solver=${caffe.model_dir}/${caffe.solver_file}' if '${exp.phase}' == 'training' else '')$",
        "$('--model=${caffe.model_dir}/${caffe.model_file}' if '${exp.phase}' == 'inference' else '')$",
        "$('-iterations ${exp.num_batches}' if '${exp.phase}' == 'inference' else '')$",
        "$('--gpu=${exp.gpus}' if '${exp.device_type}' == 'gpu' else '')$"
      ],
      "type": "str",
      "desc": "Command line arguments that launcher uses to launch Caffe."
    },
    "caffe.data_dir": {
      "val":  "",
      "type": "str",
      "desc": "A data directory if real data should be used. If empty, synthetic data is used (no data ingestion pipeline)."
    },
    "caffe.mirror": {
      "val":  true,
      "type": "bool",
      "desc": "In case of real data, specifies if 'mirrowing' should be applied."
    },
    "caffe.data_mean_file": {
      "val":  "",
      "type": "str",
      "desc": "In case of real data, specifies path to an image mean file."
    },
    "caffe.data_backend": {
      "val": "LMDB",
      "type": "str",
      "val_domain":["LMDB", "LEVELDB"],
      "desc": "In case of real data, specifies its storage backend ('LMDB' or 'LEVELDB')."
    },
    "caffe.host_path": {
      "val": "${${caffe.fork}_caffe.host_path}",
      "type": "str",
      "desc": "Path to a caffe executable in case of bare metal run."
    },
    "caffe.docker_image": {
      "val": "${${caffe.fork}_caffe.docker_image}",
      "type": "str",
      "desc": "The name of a docker image to use."
    },
    "caffe.docker_args": {
      "val": [
        "-i",
        "--security-opt seccomp=unconfined",
        "--pid=host",
        "--volume=${DLBS_ROOT}/models/${exp.model}:/workspace/model",
        "$('--volume=${runtime.cuda_cache}:/workspace/cuda_cache' if '${runtime.cuda_cache}' else '')$",
        "$('--volume=${caffe.data_dir}:/workspace/data' if '${caffe.data_dir}' else '')$",
        "$('--volume=${monitor.pid_folder}:/workspace/tmp' if ${monitor.frequency} > 0 else '')$",
        "${exp.docker_args}",
        "${caffe.docker_image}"
      ],
      "type": "str",
      "desc": "In case if containerized benchmarks, this are the docker parameters."
    },
    "bvlc_caffe.host_path": {
      "val": "${HOME}/projects/bvlc_caffe/build/tools",
      "type": "str",
      "desc": "Path to a BVLC Caffe executable in case of a bare metal run."
    },
    "bvlc_caffe.host_libpath": {
      "val": "",
      "type": "str",
      "desc": "Basically, it's a LD_LIBRARY_PATH for BVLC Caffe in case of a bare metal run."
    },
    "bvlc_caffe.docker_image": {
      "val": "hpe/bvlc_caffe:cuda9-cudnn7",
      "type": "str",
      "desc": "The name of a docker image to use for BVLC Caffe."
    },
    "nvidia_caffe.host_path": {
      "val": "${HOME}/projects/nvidia_caffe/build/tools",
      "type": "str",
      "desc": "Path to a NVIDIA Caffe executable in case of a bare metal run."
    },
    "nvidia_caffe.host_libpath": {
      "val": "",
      "type": "str",
      "desc": "Basically, it's a LD_LIBRARY_PATH for NVIDIA Caffe in case of a bare metal run."
    },
    "nvidia_caffe.docker_image": {
      "val": "hpe/nvidia_caffe:cuda9-cudnn7",
      "type": "str",
      "desc": "The name of a docker image to use for NVIDIA Caffe."
    },
    "nvidia_caffe.precision":{
      "val": "$('float32' if '${exp.dtype}' in ('float', 'float32', 'int8') else 'float16')$",
      "type": "str",
      "val_domain": ["float32", "float16", "mixed"],
      "desc": [
        "Parameter that specifies what components in NVIDIA Caffe use what precision:",
        "    float32   Use FP32 for training values storage and matrix-mult accumulator.",
        "              Use FP32 for master weights.",
        "    float16   Use FP16 for training values storage and matrix-mult accumulator",
        "              Use FP16 for master weights.",
        "    mixed     Use FP16 for training values storage and FP32 for matrix-mult accumulator",
        "              Use FP32 for master weights.",
        "More fine-grained control over these values can be done by directly manipulating",
        "the following parameters: ",
        "    nvidia_caffe.solver_precision          Master weights",
        "    nvidia_caffe.forward_precision         Training values storage",
        "    nvidia_caffe.backward_precision        Training values storage",
        "    nvidia_caffe.forward_math_precision    Matrix-mult accumulator",
        "    nvidia_caffe.backward_math_precision   Matrix-mult accumulator",
        "Default value depends on exp.dtype parameters:",
        "    exp.dtype == float32 -> nvidia_caffe.precision = float32",
        "    exp.dtype == float16 -> nvidia_caffe.precision = float16",
        "    exp.dtype == int32   -> nvidia_caffe.precision is set to float32 and experiment will notbe ran",
        "For information:",
        "    http://on-demand.gputechconf.com/gtc/2017/presentation/s7218-training-with-mixed-precision-boris-ginsburg.pdf",
        "    https://github.com/NVIDIA/caffe/issues/420"
      ]
    },
    "nvidia_caffe.solver_precision": {
      "val": "$('FLOAT' if '${nvidia_caffe.precision}' in ('float32', 'mixed') else 'FLOAT16')$",
      "type": "str",
      "val_domain": ["FLOAT", "FLOAT16"],
      "desc": [
        "Precision for a solver (FLOAT, FLOAT16). Only for NVIDIA Caffe.",
        "More details are here: http://on-demand.gputechconf.com/gtc/2017/presentation/s7218-training-with-mixed-precision-boris-ginsburg.pdf"
      ]
    },
    "nvidia_caffe.forward_precision": {
      "val": "$('FLOAT16' if '${nvidia_caffe.precision}' in ('float16', 'mixed') else 'FLOAT')$",
      "type": "str",
      "val_domain": ["FLOAT", "FLOAT16"],
      "desc": [
        "Precision for a forward pass (FLOAT, FLOAT16). Only for NVIDIA Caffe.",
        "More details are here: http://on-demand.gputechconf.com/gtc/2017/presentation/s7218-training-with-mixed-precision-boris-ginsburg.pdf"
      ]
    },
    "nvidia_caffe.backward_precision": {
      "val": "$('FLOAT16' if '${nvidia_caffe.precision}' in ('float16', 'mixed') else 'FLOAT')$",
      "type": "str",
      "val_domain": ["FLOAT", "FLOAT16"],
      "desc": [
        "Precision for a backward pass (FLOAT, FLOAT16). Only for NVIDIA Caffe.",
        "More details are here: http://on-demand.gputechconf.com/gtc/2017/presentation/s7218-training-with-mixed-precision-boris-ginsburg.pdf"
      ]
    },
    "nvidia_caffe.forward_math_precision": {
      "val": "$('FLOAT' if '${nvidia_caffe.precision}' in ('float32', 'mixed') else 'FLOAT16')$",
      "type": "str",
      "val_domain": ["FLOAT", "FLOAT16"],
      "desc": [
        "Precision for a forward math (FLOAT, FLOAT16). Only for NVIDIA Caffe.",
        "More details are here: http://on-demand.gputechconf.com/gtc/2017/presentation/s7218-training-with-mixed-precision-boris-ginsburg.pdf"
      ]
    },
    "nvidia_caffe.backward_math_precision": {
      "val": "$('FLOAT' if '${nvidia_caffe.precision}' in ('float32', 'mixed') else 'FLOAT16')$",
      "type": "str",
      "val_domain": ["FLOAT", "FLOAT16"],
      "desc": [
        "Precision for a backward math (FLOAT, FLOAT16). Only for NVIDIA Caffe.",
        "More details are here: http://on-demand.gputechconf.com/gtc/2017/presentation/s7218-training-with-mixed-precision-boris-ginsburg.pdf"
      ]
    },
    "intel_caffe.host_path": {
      "val": "${HOME}/projects/intel_caffe/build/tools",
      "type": "str",
      "desc": "Path to an Intel Caffe executable in case of a bare metal run."
    },
    "intel_caffe.host_libpath": {
      "val": "",
      "type": "str",
      "desc": "Basically, it's a LD_LIBRARY_PATH for Intel Caffe in case of a bare metal run."
    },
    "intel_caffe.docker_image": {
      "val": "hpe/intel_caffe:cpu",
      "type": "str",
      "desc": "The name of a docker image to use for Intel Caffe."
    }
  },
  "extensions": [
    {
      "condition":  { "exp.framework": "bvlc_caffe" },
      "parameters": { "exp.framework_title":"BVLC Caffe", "exp.framework_family": "caffe", "caffe.fork": "bvlc" }
    },
    {
      "condition":  { "exp.framework": "nvidia_caffe" },
      "parameters": { "exp.framework_title":"NVIDIA Caffe", "exp.framework_family": "caffe", "caffe.fork": "nvidia" }
    },
    {
      "condition":  { "exp.framework": "intel_caffe" },
      "parameters": { "exp.framework_title":"Intel Caffe", "exp.framework_family": "caffe", "caffe.fork": "intel" }
    },
    {
      "condition":  { "exp.framework": ["bvlc_caffe", "nvidia_caffe", "intel_caffe"], "exp.docker": false },
      "parameters": {
        "caffe.env": [
          "PATH=$('${${caffe.fork}_caffe.host_path}:\\$PATH'.strip(' \t:'))$",
          "LD_LIBRARY_PATH=$('${${caffe.fork}_caffe.host_libpath}:\\$LD_LIBRARY_PATH'.strip(' \t:'))$",
          "${runtime.EXPORT_CUDA_CACHE_PATH}"
        ]
      }
    }
  ]
}
```

This configuration defines parameters that are relevant to Caffe. These parameters may depend on other Caffe parameters and other general parameters (the ones that start with `exp.`). Most of these parameters are self explanatory. The most interesting ones are the following:

1. `caffe.launcher` defines executable wrapper script that will take all these parameters and will execute Caffe translating these parameters to a Caffe format. Basically, its task is to run `caffe ...` with valid set of parameters.
2. `caffe.args` defines parameters for Caffe. One way to compute these parameters is inside `${caffe.launcher}`. In most cases it's easier to compute them in JSON config files like in this example.
3. `caffe.docker_args` defined docker parameters if containerized benchmark is requested.
4. `bvlc_caffe.host_path` defines path to Caffe executable file in case bare metal benchmark is requested.

Extensions allows to modify benchmark configuration. Several modifications can happen:

1. Add new parameters.
2. Overwrite some parameters.
3. Create multiple experiments based on one experiment.

Let's consider the first extensions

```json
{
    "condition":{ "exp.framework": "([^_]+)_caffe" },
    "parameters": {
        "exp.framework_title":"$('${__condition.exp.framework_1}'.upper() + 'Caffe')$",
        "exp.framework_family": "caffe",
        "caffe.fork": "${__condition.exp.framework_1}"
    }
}
```

Every extension has a condition. Extension is fired only when condition is satisfied. Currently, a condition can only use constant parameters/variables - those that do not depend on other variables. In this particular case, extension is fired when there exist parameter `exp.framework` with value that matches `([^_]+)_caffe` regular expression. If so, following modifications to current benchmark are applied:

1. A new value is assigned to parameter `exp.framework_title`. In this case, it's a python expression that uses special construct `${__condition.exp.framework_1}` that takes group 1 of the  matched parameter with name `exp.framework`. Group 1 in this particular case is a caffe fork written in lowercase (bvlc, nvidia, intel etc.).
2. A new value will be assigned to `exp.framework_family` parameters.
1. A new value is assigned to parameter `caffe.fork`. In this case, it's a special construct `${__condition.exp.framework_1}` that takes group 1 of the  matched parameter with name `exp.framework`. Group 1 in this case will equal to something like `bvlc`, `nvidia` or `intel`.

When all parameters are computed, the experimenter will execute `${caffe.launcher}` script and will pass to this script all defined parameters. Since in bash parameters cannot contain `.`, they will be replaced with underscores `_`. For instance, parameter `exp.replica_batch` with value 16 will be converted to `--exp_replica_batch 16`.

A wrapper shell script for Caffe looks like this:

```bash
#!/bin/bash
unknown_params_action=set
. $DLBS_ROOT/scripts/parse_options.sh || exit 1;    # Parse command line options
. $DLBS_ROOT/scripts/utils.sh
loginfo "$0 $*" >> ${exp_log_file}                  # Log command line arguments for debugging purposes
# The simulation mode: just print out what is about to be launched
if [ "$exp_status" = "simulate" ]; then
  echo "${caffe_env} ${runtime_launcher} caffe ${caffe_action} ${caffe_args}"
  exit 0
fi
# Check batch is small enough for this experiment
__batch_file__="$(dirname ${exp_log_file})/${exp_framework}_${exp_device_type}_${exp_model}.batch"
is_batch_good "${__batch_file__}" "${exp_replica_batch}" || {
  report_and_exit "skipped" "The replica batch size (${exp_replica_batch}) is too large for given SW/HW configuration." "${exp_log_file}";
}
# Make sure model exists
host_model_dir=$DLBS_ROOT/models/${exp_model}
model_file=$(find ${host_model_dir}/ -name "*.${exp_phase}.prototxt")
file_exists "$model_file" || report_and_exit "failure" "A model file ($model_file) does not exist." "${exp_log_file}"
# Copy model file and replace batch size there.
# https://github.com/BVLC/caffe/blob/master/docs/multigpu.md
# NOTE: each GPU runs the batchsize specified in your train_val.prototxt
remove_files "${host_model_dir}/${caffe_model_file}" "${host_model_dir}/${caffe_solver_file}"
cp ${model_file} ${host_model_dir}/${caffe_model_file} || {
  report_and_exit "failure" "Cannot copy \"${model_file}\" to \"${host_model_dir}/${caffe_model_file}\"" "${exp_log_file}"
}
# If we are in 'training' phase and data_dir is not empty, we need to change *.train.prototxt file here.
# Or we can have two configurations for synthetic/real data.
# Or we can specify input layers in JSON config, so that we can basically set this dynamically
if [ "${exp_phase}" == "training" ]; then
    if [ "${caffe_data_dir}" == "" ]; then
        sed -i "s/^#synthetic//g" ${host_model_dir}/${caffe_model_file}
    else
        sed -i "s/^#data//g" ${host_model_dir}/${caffe_model_file}
        sed -i "s#__CAFFE_MIRROR__#${caffe_mirror}#g" ${host_model_dir}/${caffe_model_file}
        sed -i "s#__CAFFE_DATA_MEAN_FILE__#${caffe_data_mean_file}#g" ${host_model_dir}/${caffe_model_file}
        sed -i "s#__CAFFE_DATA_DIR__#${caffe_data_dir}#g" ${host_model_dir}/${caffe_model_file}
        sed -i "s#__CAFFE_DATA_BACKEND__#${caffe_data_backend}#g" ${host_model_dir}/${caffe_model_file}
    fi
    if [ "${exp_framework}" == "nvidia_caffe" ]; then
        sed -i "s/^#precision//g" ${host_model_dir}/${caffe_model_file}
        sed -i "s/__FORWARD_TYPE___/${nvidia_caffe_forward_precision}/g" ${host_model_dir}/${caffe_model_file}
        sed -i "s/__BACKWARD_TYPE___/${nvidia_caffe_backward_precision}/g" ${host_model_dir}/${caffe_model_file}
        sed -i "s/__FORWARD_MATH___/${nvidia_caffe_forward_math_precision}/g" ${host_model_dir}/${caffe_model_file}
        sed -i "s/__BACKWARD_MATH___/${nvidia_caffe_backward_math_precision}/g" ${host_model_dir}/${caffe_model_file}
    fi
fi
# For NVIDIA Caffe, protobuf must contain effective batch size.
if [ "${caffe_fork}" == "nvidia" ]; then
    sed -i "s/__EXP_DEVICE_BATCH__/${exp_effective_batch}/g" ${host_model_dir}/${caffe_model_file}
else
    sed -i "s/__EXP_DEVICE_BATCH__/${exp_replica_batch}/g" ${host_model_dir}/${caffe_model_file}
fi
#
net_name=$(get_value_by_key "${host_model_dir}/${caffe_model_file}" "name")
[ "${exp_phase}" = "training" ] && echo -e "${caffe_solver}" > ${host_model_dir}/${caffe_solver_file}
echo "__exp.model_title__= \"${net_name}\"" >> ${exp_log_file}
# This script is to be executed inside docker container or on a host machine.
# Thus, the environment must be initialized inside this scrip lazily.
[ -z "${runtime_launcher}" ] && runtime_launcher=":;"
script="\
    export ${caffe_env};\
    echo -e \"__exp.framework_ver__= \x22\$(caffe --version | head -1 | awk '{print \$3}')\x22\";\
    echo -e \"__results.start_time__= \x22\$(date +%Y-%m-%d:%H:%M:%S:%3N)\x22\";\
    ${runtime_launcher} caffe ${caffe_action} ${caffe_args} &\
    proc_pid=\$!;\
    [ \"${monitor_frequency}\" != \"0\" ] && echo -e \"\${proc_pid}\" > ${monitor_backend_pid_folder}/proc.pid;\
    wait \${proc_pid};\
    echo -e \"__results.end_time__= \x22\$(date +%Y-%m-%d:%H:%M:%S:%3N)\x22\";\
    echo -e \"__results.proc_pid__= \${proc_pid}\";\
"

if [ "${exp_docker}" = "true" ]; then
    assert_docker_img_exists ${exp_docker_image}
    ${exp_docker_launcher} run ${caffe_docker_args} /bin/bash -c "eval $script" >> ${exp_log_file} 2>&1
else
    eval $script >> ${exp_log_file} 2>&1
fi

# Do some post-processing
remove_files "${host_model_dir}/${caffe_model_file}" "${host_model_dir}/${caffe_solver_file}"
caffe_postprocess_log "${exp_log_file}" "${__batch_file__}" "${exp_phase}" "${exp_replica_batch}" "${exp_effective_batch}" "${exp_num_batches}"
```

Its responsibility is to run caffe with parameters. It must set up environment (taking into account environmental variables), and run caffe in host or docker container depending on input parameters.

## New model
There are several possibilities.

1. If model comes with its own script, the entire script may be considered as a custom backend. See above section how to add new backend.
2. If it's just a model definition, or it's relatively easy to extract model from that project, this model can be added to existing backend. For example descriptions of existing backends,
 see section `Adding new model`:
    1. [TensorFlow](/frameworks/tensorflow.md?id=adding-new-model)
    2. [Caffe2](/frameworks/caffe2.md?id=adding-new-model)
    3. [MXNet](/frameworks/mxnet.md?id=adding-new-model)
    4. BLVC/NVIDIA/Intel [Caffe](/frameworks/caffe.md?id=adding-new-model)
