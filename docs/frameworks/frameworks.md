# __Frameworks__

Deep Learning Benchmarking Suite benchmarks frameworks via one or two intermediate scripts:

1. A shell wrapper script (`launcher`) that takes experimenter parameters and calls framework script in host or docker environment. It knows how to translate input parameters into framework specific parameters. Currently, these wrapper scripts are approximately 50-100 lines
2. A script, usually a python project, that knows how to run a framework with specified models. May not be required, like in Caffe case - launcher may directly invoke `caffe` framework. For other frameworks, we usually need a python project that for this. If possible, we take advantage of existing projects (TensorFlow case). In general, it's possible to use these frameworks without DLBS.

Navigate to specific framework section to learn more about integration:

* [Caffe](/frameworks/caffe.md?id=caffe)
* [Caffe2](/frameworks/caffe2.md?id=caffe2)
* [MXNet](/frameworks/mxnet.md?id=mxnet)
* [TensorFlow](/frameworks/tensorflow.md?id=tensorflow)
