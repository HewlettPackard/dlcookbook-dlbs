# OpenVINO benchmark backend with custom MKLDNN plugin

> Work in progress. DLBS now supports OpenVINO benchmark backend in experimental regime.

> The [Dockerfile](./Dockerfile) needs re-design and may not work in all cases, in particular, if your environment does not define `http_proxy` and/or `https_proxy` variables, you will need to comment line 12 in that file:
```
RUN git config --global http.proxy ${http_proxy} && git config --global https.proxy ${https_proxy}
``` 

This docker image serves the purpose of running benchmarks with and without VNNI instructions on systems with CPUs that support them.  
By default, if VNNI instructions are avaialble, they will be used and there is no way to disable that programmatically. The workaround (thanks to our partners from Intel who helped us with this) is to patch mkldnn library and rebuild the MKLDNN CPU pluging.  
The algorithm is following:
1. First, you need to build a [`dlbs/openvino:19.09`](../19.09) image with OpenVINO inference engine and benchmark app.
2. Then build this image that depends on 19.09 version. The build process is the following (see [Dockerfile](./Dockerfile) in this folder):
   - Download a patched version of open source version of OpenVINI inference engine from my (Sergey Serebryakov) repository, it is located [here](https://github.com/sergey-serebryakov/dldt/tree/2019_R3_DLBS) in branch `2019_R3_DLBS`.
   - Build MKLDNN plugin and replace existing plugin with this new one.


The environmental variable `MKLDNN_NO_VNNI` controls new functionality. By defautl, if it is not defined, VNNI will be used. Set it to 1 to disable VNNI if they are available:
export MKLDNN_NO_VNNI=1.  
If OpenVINO benchmark backend is used with DLBS, then use `runtime.launcher` to provide this variable:
```json
{
  "parameters": {
    "runtime.launcher": "MKLDNN_NO_VNNI=1"
  }
}
```
or
```bash
python experimenter.py --config ./config.json -Pruntime.launcher='"MKLDNN_NO_VNNI=1"'
```
To double check that you are using the right version, make sure you see the output that looks like this:
```
MKLDNN_NO_VNNI: 1 (if 0: use VNNI if available; if 1: do not use VNNI even if available)
```
For more details and advanced examples, study the [config_vnni_tests.json](../../../tutorials/recipes/openvino/config_vnni_tests.json) configuration file. 

## Technical details
Modifications were only required in one function [here](https://github.com/sergey-serebryakov/dldt/blob/4c685991c454bd038368aa985c66a819f2303523/inference-engine/thirdparty/mkl-dnn/src/cpu/cpu_isa_traits.hpp#L113).