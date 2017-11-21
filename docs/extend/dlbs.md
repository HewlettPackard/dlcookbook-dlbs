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
        "caffe.launcher": "${DLBS_ROOT}/scripts/launchers/caffe.sh",
        "caffe.fork": "bvlc",
        "caffe.phase": "$('train' if '${exp.phase}' == 'training' else 'deploy')$",
        "caffe.action": "$('train' if '${exp.phase}' == 'training' else 'time')$",
        "caffe.model_file": "${exp.id}.model.prototxt",
        "caffe.solver_file": "${exp.id}.solver.prototxt",
        "caffe.model_dir": "$('${DLBS_ROOT}/models/${exp.model}' if '${exp.env}' == 'host' else '/workspace/model')$",
        "caffe.solver": [
            "net: '${caffe.model_dir}/${caffe.model_file}'\\n",
            "max_iter: ${exp.bench_iters}\\n",
            "test_interval: 0\\n",
            "snapshot: 0\\n",
            "snapshot_after_train: false\\n",
            "base_lr: 0.01\\n",
            "lr_policy: 'fixed'\\n",
            "solver_mode: $('${exp.device}'.upper())$\\n",
            "$('solver_data_type: ${nvidia_caffe.solver_precision}' if '${exp.framework_id}' == 'nvidia_caffe' else '')$"
        ],
        "caffe.args": [
            "$('--solver=${caffe.model_dir}/${caffe.solver_file}' if '${exp.phase}' == 'training' else '')$",
            "$('--model=${caffe.model_dir}/${caffe.model_file}' if '${exp.phase}' == 'inference' else '')$",
            "$('-iterations ${exp.bench_iters}' if '${exp.phase}' == 'inference' else '')$",
            "$('--gpu=${exp.gpus}' if '${exp.device}' == 'gpu' else '')$"
        ],
        "caffe.data_dir": "",
        "caffe.mirror": true,
        "caffe.data_mean_file": "",
        "caffe.data_backend": "LMDB",
        "caffe.host.path": "${${caffe.fork}_caffe.host.path}",
        "caffe.docker.image": "${${caffe.fork}_caffe.docker.image}",
        "caffe.docker.args": [
            "-i",
            "--security-opt seccomp=unconfined",
            "--pid=host",
            "--volume=${DLBS_ROOT}/models/${exp.model}:/workspace/model",
            "$('--volume=${CUDA_CACHE_PATH}:/workspace/cuda_cache' if '${CUDA_CACHE_PATH}' else '')$",
            "$('--volume=${caffe.data_dir}:/workspace/data' if '${caffe.data_dir}' else '')$",
            "$('--volume=${resource_monitor.pid_file_folder}:/workspace/tmp' if '${resource_monitor.enabled}' == 'true' else '')$",
            "${caffe.docker.image}"
        ],
        "bvlc_caffe.host.path": "${HOME}/projects/bvlc_caffe/build/tools",
        "bvlc_caffe.host.libpath": "",
        "bvlc_caffe.docker.image": "hpe/bvlc_caffe:cuda8-cudnn7",
        "nvidia_caffe.host.path": "${HOME}/projects/nvidia_caffe/build/tools",
        "nvidia_caffe.host.libpath": "",
        "nvidia_caffe.docker.image": "hpe/nvidia_caffe:cuda8-cudnn7",
        "nvidia_caffe.solver_precision": "FLOAT",
        "nvidia_caffe.forward_precision": "FLOAT",
        "nvidia_caffe.forward_math_precision": "FLOAT",
        "nvidia_caffe.backward_precision": "FLOAT",
        "nvidia_caffe.backward_math_precision": "FLOAT",
        "intel_caffe.host.path": "${HOME}/projects/intel_caffe/build/tools",
        "intel_caffe.host.libpath": "",
        "intel_caffe.docker.image": "hpe/intel_caffe:cpu"
    },
    "extensions": [
        {
            "condition":{ "exp.framework": "([^_]+)_caffe" },
            "parameters": {
                "exp.framework_id":"${__condition.exp.framework_0}",
                "exp.framework": "caffe",
                "caffe.fork": "${__condition.exp.framework_1}"
            }
        },
        {
            "condition":{ "exp.framework": "caffe", "exp.env": "docker" },
            "parameters": { "caffe.env": ["$('CUDA_CACHE_PATH=${runtime.cuda_cache}' if '${CUDA_CACHE_PATH}' else '')$"]}
        },
        {
            "condition":{ "exp.framework": "caffe", "exp.env": "host" },
            "parameters": {
                "caffe.env": [
                    "PATH=$('${${caffe.fork}_caffe.host.path}:\\$PATH'.strip(' \t:'))$",
                    "LD_LIBRARY_PATH=$('${${caffe.fork}_caffe.host.libpath}:\\$LD_LIBRARY_PATH'.strip(' \t:'))$",
                    "$('CUDA_CACHE_PATH=${runtime.cuda_cache}' if '${CUDA_CACHE_PATH}' else '')$"
                ]
            }
        }
    ]
}
```

This configuration defines parameters that are relevant to Caffe. These parameters may depend on other Caffe parameters and other general parameters (the ones that start with `exp.`). Most of these parameters are self explanatory. The most interesting ones are the following:

1. `caffe.launcher` defines executable wrapper script that will take all these parameters and will execute Caffe translating these parameters to a Caffe format. Basically, its task is to run `caffe ...` with valid set of parameters.
2. `caffe.args` defines parameters for Caffe. One way to compute these parameters is inside `${caffe.launcher}`. In most cases it's easier to compute them in JSON config files like in this example.
3. `caffe.docker.args` defined docker parameters if containerized benchmark is requested.
4. `bvlc_caffe.host.path` defines path to Caffe executable file in case bare metal benchmark is requested.

Extensions allows to modify benchmark configuration. Several modifications can happen:

1. Add new parameters.
2. Overwrite some parameters.
3. Create multiple experiments based on one experiment.

Let's consider the first extensions

```json
{
    "condition":{ "exp.framework": "([^_]+)_caffe" },
    "parameters": {
        "exp.framework_id":"${__condition.exp.framework_0}",
        "exp.framework": "caffe",
        "caffe.fork": "${__condition.exp.framework_1}"
    }
}
```

Every extension has a condition. Extension is fired only when condition is satisfied. In this particular case extension is fired when there exist parameter `exp.framework` with value that matches `([^_]+)_caffe` regular expression. If so, following modifications to current benchmark are applied:

1. A new value is assigned to parameter `exp.framework_id`. In this case, it's a special construct `${__condition.exp.framework_0}` that takes group 0 of the  matched parameter with name `exp.framework`. Group 0 is a whole match, so it will equal to `exp.framework` parameter value.
2. A new value will be assigned to `exp.framework` parameters.
1. A new value is assigned to parameter `caffe.fork`. In this case, it's a special construct `${__condition.exp.framework_1}` that takes group 1 of the  matched parameter with name `exp.framework`. Group 1 in this case will equal to something like `bvlc`, `nvidia` or `intel`.

When all parameters are computed, the experimenter will execute `${caffe.launcher}` script and will pass to this script all defined parameters. Since in bash parameters cannot contain `.`, they will be replaced with underscores `_`. For instance, parameter `exp.device_batch` with value 16 will be converted to `--exp_device_batch 16`.

A wrapper shell script for Caffe looks like this:

```bash
#!/bin/bash
unknown_params_action=set
. $DLBS_ROOT/scripts/parse_options.sh || exit 1;    # Parse command line options
. $DLBS_ROOT/scripts/utils.sh
loginfo "$0 $*" >> ${exp_log_file}                  # Log command line arguments for debugging purposes
# The simulation mode: just print out what is about to be launched
if [ "$exp_simulation" = "true" ]; then
    echo "${runtime_bind_proc} caffe ${caffe_action} ${caffe_args}"
    exit 0
fi
__framework__="$(echo ${caffe_fork} | tr '[:lower:]' '[:upper:]')_Caffe"
# Check batch is small enough for this experiment
__batch_file__="$(dirname ${exp_log_file})/${exp_framework_id}_${exp_device}_${exp_model}.batch"
is_batch_good "${__batch_file__}" "${exp_device_batch}" || { logwarn "Device batch is too big for configuration"; exit 0; }

# Make sure model exists
host_model_dir=$DLBS_ROOT/models/${exp_model}
model_file=$(find ${host_model_dir}/ -name "*.${exp_phase}.prototxt")
file_exists "$model_file" || { logerr "model file does not exist"; exit 1; }

# Copy model file and replace batch size there.
# https://github.com/BVLC/caffe/blob/master/docs/multigpu.md
# NOTE: each GPU runs the batchsize specified in your train_val.prototxt
remove_files "${host_model_dir}/${caffe_model_file}" "${host_model_dir}/${caffe_solver_file}"
cp ${model_file} ${host_model_dir}/${caffe_model_file} || logfatal "Cannot cp \"${model_file}\" to \"${host_model_dir}/${caffe_model_file}\""
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
    if [ "${exp_framework_id}" == "nvidia_caffe" ]; then
        sed -i "s/^#precision//g" ${host_model_dir}/${caffe_model_file}
        sed -i "s/__FORWARD_TYPE___/${nvidia_caffe.forward_precision}/g" ${host_model_dir}/${caffe_model_file}
        sed -i "s/__BACKWARD_TYPE___/${nvidia_caffe.backward_precision}/g" ${host_model_dir}/${caffe_model_file}
        sed -i "s/__FORWARD_MATH___/${nvidia_caffe.forward_math_precision}/g" ${host_model_dir}/${caffe_model_file}
        sed -i "s/__BACKWARD_MATH___/${nvidia_caffe.backward_math_precision}/g" ${host_model_dir}/${caffe_model_file}
    fi
fi
sed -i "s/__EXP_DEVICE_BATCH__/${exp_device_batch}/g" ${host_model_dir}/${caffe_model_file}
#
net_name=$(get_value_by_key "${host_model_dir}/${caffe_model_file}" "name")
[ "${exp_phase}" = "training" ] && echo -e "${caffe_solver}" > ${host_model_dir}/${caffe_solver_file}
echo "__exp.model_title__= \"${net_name}\"" >> ${exp_log_file}
echo "__exp.framework_title__= \"${__framework__}\"" >> ${exp_log_file}
# This script is to be executed inside docker container or on a host machine.
# Thus, the environment must be initialized inside this scrip lazily.
[ -z "${runtime_limit_resources}" ] && runtime_limit_resources=":;"
script="\
    export ${caffe_env};\
    echo -e \"__caffe.version__= \x22\$(caffe --version | head -1 | awk '{print \$3}')\x22\";\
    echo -e \"__results.start_time__= \x22\$(date +%Y-%m-%d:%H:%M:%S:%3N)\x22\";\
    ${runtime_limit_resources}\
    ${runtime_bind_proc} caffe ${caffe_action} ${caffe_args} &\
    proc_pid=\$!;\
    [ "${resource_monitor_enabled}" = "true" ] && echo -e \"\${proc_pid}\" > ${resource_monitor_pid_file_folder}/proc.pid;\
    wait \${proc_pid};\
    echo -e \"__results.end_time__= \x22\$(date +%Y-%m-%d:%H:%M:%S:%3N)\x22\";\
    echo -e \"__results.proc_pid__= \${proc_pid}\";\
"

if [ "${exp_env}" = "docker" ]; then
    assert_docker_img_exists ${caffe_docker_image}
    ${exp_docker_launcher} run ${caffe_docker_args} /bin/bash -c "eval $script" >> ${exp_log_file} 2>&1
else
    eval $script >> ${exp_log_file} 2>&1
fi

# Do some post-processing
remove_files "${host_model_dir}/${caffe_model_file}" "${host_model_dir}/${caffe_solver_file}"
caffe_postprocess_log "${exp_log_file}" "${__batch_file__}" "${exp_phase}" "${exp_device_batch}" "${exp_effective_batch}" "${exp_bench_iters}"
```

Its responsibility to run caffe with parameters. It must set up environment (taking into account environmental variables), and run caffe in host or docker container depending on input parameters.

## New model
There are several possibilities.

1. If model comes with its own script, the entire script may be considered as a custom backend. See above section how to add new backend.
2. If it's just a model definition, or it's relatively easy to extract model from that project, ths model can be added to existing backend. For example descriptions of existing backends,
 see section `Adding new model`:
    1. [TensorFlow](/frameworks/tensorflow.md?id=adding-new-model)
    2. [Caffe2](/frameworks/caffe2.md?id=adding-new-model)
    3. [MXNet](/frameworks/mxnet.md?id=adding-new-model)
    4. BLVC/NVIDIA/Intel [Caffe](/frameworks/caffe.md?id=adding-new-model)
