# Building TensorRT benchmark backend

The benchmark tool uses cmake. In particular, you will need cmake, CUDA, boost program options, opencv 2 and TensorRT 3.0.4
libraries installed in your system. Go to `${DLBS_ROOT}/src/tensorrt` and run the following commands:
```bash
mkdir ./build && cd ./build
cmake .. && make -j$(nproc)
```

Several cmake configuration parameters affect the building process:

1. `HOST_DTYPE=[INT8]` A data type used to store data in host OS. Two options are supported - unsigned char (INT8)
   and float (SP32). By default, INT8 is used. To use float data type instead, provide the following:
   ```bash
       cmake -DHOST_DTYPE=SP32 ..
   ```

2. `DEBUG_LOG=[OFF]` Enables/disables detailed logging. May be useful for debugging purposes:
   ```bash
       cmake -DDEBUG_LOG=ON ..
   ```

Some of the standard parameters:
1. `CMAKE_BUILD_TYPE=[Release]` Build type. By default, release binaries are build. Set to `Debug` to debug the tool.

### Building documentation
Documentation is built with doxygen. You will need `doxygen` and `graphviz` i.e.:
   ```bash
       sudo apt-get install doxygen graphviz
   ```

To build documentation separately, build `build_docs` target after configuring the project:
   ```bash
       make build_docs
   ```
In the container, the documentation will be installed to `/opt/tensorrt/share/doc/html` (index.html).