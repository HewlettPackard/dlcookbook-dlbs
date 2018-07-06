#!/bin/bash
python ./test_variable_expansion.py  --bench_root `pwd`  --dlbs_root /lvol/sfleisch/dlbs -u "/lvol/sfleisch/dlbs/benchmarks/nvcnn_hvd_singularity.json"  -p exp.gpufreq='"'4'"'

