# TensorRT benchmark backend docker images


### 18.08 / 18.10
Docker images that use the same TensorRT 3.0.4 and the same benchmark backend.
Differences are as follows:

 - `18.08`
    - Uses float data type to store images in host memory
    - Uses standard file reader that can instruct OS to not cache read files.
 - `18.10`
    - Uses unsigned char data type to store images in host memory
    - Uses direct io file reader that reads data bypassing OS caches (good for
      storage benchmarks).

These differences are either compile time configuration (like data type) or
run rime configuration (via environment variables) like type of file reader.

We keep these two versions now for historical reasons. We plan to remove 18.08
configuration in the future. This documentation will be updated accordingly.
