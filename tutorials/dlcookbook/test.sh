#!/bin/bash

export BENCH_ROOT=$( cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )
export CUDA_CACHE_PATH=/dev/shm/cuda_cache
. ${BENCH_ROOT}/../../scripts/environment.sh
script=$DLBS_ROOT/python/dlbs/experimenter.py


python $script validate --log-level=info\
                    -Pexp.env='"host"'\
                    -Pexp.framework='"tensorflow"'\
                    -Ptensorflow.var_update='"replicated"'\
                    -Ptensorflow.local_parameter_device='"cpu"'\
                    -Ptensorflow.data_dir='"/dev/shm/tfrecords"'\
                    -Pexp.phase='"training"'\
                    -Vexp.gpus='["0", "0,1", "0,1,2,3"]'\
                    -Pexp.bench_iters='150'\
                    -Pexp.log_file='"${BENCH_ROOT}/data/${exp.framework_id}/${exp.env}/${exp.phase}/${exp.device}/$(\"${exp.gpus}\".replace(\",\",\".\"))$_${exp.model}_${exp.effective_batch}.log"'\
                    -Vexp.model='["alexnet", "googlenet", "resnet50", "resnet101", "resnet152", "vgg16", "vgg19"]'\
                    -Pbvlc_caffe.host.libpath="\"${host_libpath}\""\
                    -Pnvidia_caffe.host.libpath="\"${host_libpath}\""\
                    -Pmxnet.host.libpath="\"${host_libpath}\""\
                    -Pbvlc_caffe.host.libpath="\"${host_libpath}\""\
                    -Pnvidia_caffe.host.libpath="\"${host_libpath}\""\
                    -Ptensorflow.host.libpath="\"${tf_host_libpath}\""\
                    -E'{"condition":{"exp.model":["googlenet","alexnet"]},"cases":[{"exp.device_batch":64},{"exp.device_batch":128},{"exp.device_batch":256},{"exp.device_batch":512}]}'\
                    -E'{"condition":{"exp.model":["resnet50", "vgg16", "vgg19"]},"cases":[{"exp.device_batch":32},{"exp.device_batch":64},{"exp.device_batch":128}]}'\
                    -E'{"condition":{"exp.model":["resnet101", "resnet152"]},"cases":[{"exp.device_batch":16},{"exp.device_batch":32},{"exp.device_batch":64}]}'
