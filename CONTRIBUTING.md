# Contributing to Deep Learning Benchmarking Suite
We'd like to accept your contributions - both bug fixes and new features! All contributions must include acceptance of the [DCO](https://developercertificate.org/):

> Developer Certificate of Origin  
> Version 1.1
>
> Copyright (C) 2004, 2006 The Linux Foundation and its contributors.  
> 1 Letterman Drive  
> Suite D4700  
> San Francisco, CA, 94129  
>  
> Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.  
>  
> Developer's Certificate of Origin 1.1
>
> By making a contribution to this project, I certify that:
>
> (a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or
>
> (b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or
>  
>  (c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.
>  
>  (d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.

To accept the DCO, simply add this line to each commit message with your name and email address:
```bash
Signed-off-by: Project Contributor <project@contributor.com>
```
For legal reasons, no anonymous or pseudonymous contributions are accepted.

If you have `user.name` and `user.email` variables in your git config defined, simply use `-s` switch `git commit -s`.


### Project structure
- `python/dlbs` Core of Benchmarking Suite.
- `python/caffe2_benchmarks` A benchmark backend for Caffe2.
- `python/mxnet_benchmarks` A benchmark backend for MXNET.
- `python/pytorch_benchmarks` A benchmark backend for PyTorch.
- `python/tf_cnn_benchmarks` A benchmark backend for TensorFlow from Google. Please, send your enhancements/bug fixes [here](https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks).
- `python/nvcnn_benchmarks` A benchmark backend for TensorFlow from NVIDIA. This is a file from one of the TensorFlow docker images from NVIDIA GPU Cloud.
- `python/nvtfcnn_benchmarks` A benchmark backend for TensorFlow from NVIDIA. This is a file from one of the TensorFlow docker images from NVIDIA GPU Cloud.
- `src/tensorrt` A benchmark backend for TensorRT.

### Writing documentation
- For python projects, DLBS uses (will use) Google [style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) formatting.
- For C++ projects, DLBS uses [Doxygen](http://www.doxygen.nl/).

### Checking your code
Use `pylint` to check your code updates.
