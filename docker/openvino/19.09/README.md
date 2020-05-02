# OpenVINO benchmark backend

> Work in progress. DLBS now supports OpenVINO benchmark backend in experimental regime.
 
This version of the OpenVINO runtime depends on the following OpenVINO package (it is defined in this [file](../../versions)):
```json
{"openvino/19.09": "l_openvino_toolkit_p_2019.2.275"}
```

## Configuring environment
If required, make sure your environment provides information about proxy servers:
```bash
export http_proxy=http://YOUR_HTTP_PROXY_SERVER:PORT
export https_proxy=https://YOUR_HTTP_PROXY_SERVER:PORT
```

## Building docker image
To build the [OpenVINO docker image](./Dockerfile), it should be sufficient to run the following command:
```bash
./build.sh --prefix dlbs ./openvino/19.09
``` 
from this [folder](../../). The build script downloads the required package automatically. If that somehow does not work, download this package manually. Run it from this folder:
```bash
wget http://registrationcenter-download.intel.com/akdlm/irc_nas/15792/l_openvino_toolkit_p_2019.2.275.tgz
tar -xf l_openvino_toolkit*
```
and then, build the docker image as described above.

## Running example benchmarks
Go to this [folder](../../../tutorials/recipes/openvino) and [this](../../../tutorials/recipes/openvino/run) script. Once done, search for logs in a `log` sub-directory.

## Technical details
DLBS uses this [launcher script](../../../scripts/launchers/openvino.sh) to run benchmarks with OpenVINO backed. It runs the following sequence of commands:
```bash
# Setup the environment
source \${OPENVINO_DIR}/bin/setupvars.sh

# Download a model
cd ${OPENVINO_DIR}/deployment_tools/open_model_zoo/tools/downloader
python ./downloader.py --name MODEL_NAME --output_dir /mnt/openvino/models --cache_dir /mnt/openvino/cache

# Run benchmarks
cd /opt/intel/openvino_benchmark_app/intel64/Release
./benchmark_app -m MODEL_FILE -api OPENVINO_API -b REPLICA_BATCH_SIZE
```
Other potential parameters for the benchmark app are described [here](https://docs.openvinotoolkit.org/latest/_inference_engine_samples_benchmark_app_README.html). DLBS configures OpenVINO benchmark backend in this [file](../../../python/dlbs/configs/openvino.json)

On the first run, DLBS creates a cache folder `${HOME}/.dlbs/openvino`. In that folder, a file (`list_topologies.yml`) lists available models. 

## Benchmarking VNNI instructions
The [config_vnni_tests.json](./config_vnni_tests.json) defines a test configuration for studying the impact of VNNI instructions on inference workloads. To use that, you need:
1. Build 'dlbs/openvino:19.09' docker image
2. Build 'dlbs/openvino:19.09-custom-mkldnn' docker image
in this particular order. For more technical details, read this [README](../../../docker/openvino/19.09-custom-mkldnn/README.md) file.
