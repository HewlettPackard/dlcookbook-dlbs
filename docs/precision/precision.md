# __Data Precision__
Deep Learning Benchmarking Suite supports training/inference phases with different
data types (precision) - `float32` (single precision), `float16` (half precision)
and `int8` - (integer 8) that mostly used for inference in TensorRT. Support of different
data types slightly varies from one framework to another.

At very high level, there are two parameters affecting data type:

1. `exp.dtype` Defines precision at high level. Possible values are `float32`,
   `float16` and `int8`. The `float` value is an alias for `float32`.
2. `exp.use_tensor_core` Enables tensor core math under certain general conditions:
   1. NVIDA Volta GPUs
   2. CUDA 9.0
   3. cuDNN 7.

### BVLC/Intel Caffe
No support at this point for `half` (float16) and `int8` (int8) precision.

### NVIDIA Caffe
NVIDIA Caffe supports float32 and float16 data types. There are three scenarios:

1. Single precision mode. Everything is stored as FP32.
2. Half precision mode. Everything is stored as FP16.
3. Mixed precision mode. Master gradients and math are FP32 (tensor core),
   forward/backward pass is done in FP16.

For more details see this [thread](https://github.com/NVIDIA/caffe/issues/420) and
this [presentation](http://on-demand.gputechconf.com/gtc/2017/presentation/s7218-training-with-mixed-precision-boris-ginsburg.pdf).

The logic is the following:
1. User can control NVIDIA Caffe precision mode with `exp.dtype` parameters. If it is
   float32, NVIDIA Caffe will operate in single precision mode. If it is float16, NVIDIA
   Caffe will operate in half precision mode.
2. User can override this default behavior by providing `nvidia_caffe.precision` parameter:
   1. `float32` Use FP32 for training values storage and matrix-mult accumulator, use FP32
      for master weights.
   2. `float16` Use FP16 for training values storage and matrix-mult accumulator, use FP16
      for master weights.
   3. `mixed` Use FP16 for training values storage and FP32 for matrix-mult accumulator, use
      FP32 for master weights.
3. An even more fine-grained control can be achieved by assigning FLOAT or FLOAT16 values to
   low level parameters:
   1. `nvidia_caffe.solver_precision` Master weights.
   2. `nvidia_caffe.forward_precision` Training values storage.
   3. `nvidia_caffe.backward_precision` Training values storage.
   4. `nvidia_caffe.forward_math_precision` Matrix-mult accumulator.
   5. `nvidia_caffe.backward_math_precision` Matrix-mult accumulator.

See the above mentioned links to learn what these parameters affect. For Volta GPUs it is
recommended to use `nvidia_caffe.precision` parameter with value `mixed` i.e.:
```bash
python experimenter.py ... --Pnvidia_caffe.precision=`"mixed"` ...
```
The `exp.use_tensor_core` does not affect behavior of NVIDIA Caffe at this point.

### TensorRT
TensorRT supports single, half and int8 inference. Use `exp.dtype` to control it.

### MXNet
The MXNet framework supports float32/float16 with optional tensor core math. DL Benchmarking
Suite will set up the environment. Use standard parameters `exp.dtype` and `exp.use_tensor_core`
to specify benchmark settings. The data type can be either float32 or float16.
The tensor core math is controlled via environmental variable MXNET_CUDA_ALLOW_TENSOR_CORE. See
this [code snippet](https://github.com/apache/incubator-mxnet/blob/a36bf573ad82550dbb6692a89d7ddd1d5e4487fd/src/common/cuda_utils.h)
how it is used. This environmental variable will be set automatically by benchmarking suite.

### Caffe2
The Caffe2 framework supports float32/float16 data types, tensor core operations and
FP16 compute (FP16 SGD - we do not currently support it but will in the future).
Use standard parameters `exp.dtype` and `exp.use_tensor_core` to specify benchmark
settings. As a reference implementation I used this ResNet50 [trainer](https://github.com/caffe2/caffe2/blob/master/caffe2/python/examples/resnet50_trainer.py).

### TensorFlow
The version of TensorFlow backend we use now (tf_cnn_benchmarks) supports single/half
precision benchmarks. This is controlled by a standard parameter `exp.dtype`. When user
disables tensor ops with `exp.use_tensor_core=false`, DLBS exports TF_DISABLE_CUDNN_TENSOR_OP_MATH
envirnmental variable with value 'false'.

### PyTorch
PyTorch accepts standard `exp.dtype` parameter and supports single and half precision
data types.
