This folder contains python scripts that are used during benchmarking processes for various purposes. It is a good idea to place here general purpose scripts that may be useful in various scenarios. This folder is added to PYTHONPATH so the scrips are available by their names.

Most top level files are well documented. So, the best way to figure out what they do is to study the source code.

### caffe2tf: convert models from Caffe to TensorFlow format.
The code is based on on this [project](https://github.com/ethereon/caffe-tensorflow) and is included under the `kaffe` directory. It was slightly modified to work correctly with latest TensorFlow. The tool converts models from Caffe format to TensorFlow. TensorFlow format is a python based custom model definition. TensorFlow models after the conversion can only be used with code that resides in `kaffe` directory. Models must be named as `ModelName.kaffe.py` so that benchmarking scripts can work with them. The tool can convert not only model definitions (`*.prototxt` -> `*kaffe.py`) but also models snapshots.

For example, to convert Caffe's network `alexnet.deploy.prototxt` into a TensorFlow graph definition
stored in file name `alexnet.kaffe.py` use the following command:
```python
python caffe2tf.py --caffedef alexnet.deploy.prototxt --tfdef alexnet.kaffe.py
```

To convert Caffe's network `alexnet.deploy.prototxt` and associated model
`alexnet.caffemodel` into a TensorFlow graph definition stored in file name
`alexnet.kaffe.py` and weights stored in file `alexnet.kaffe.npy` use the following comamnd:
```python
python caffe2tf.py --caffedef alexnet.deploy.prototxt --caffemodel alexnet.caffemodel
                  --tfdef alexnet.kaffe.py --tfmodel alexnet.kaffe.npy
```

### time_tf: benchmark TensorFlow graphs
This tool is used to benchmark TensorFlow graphs. It is based on this [implementation](https://github.com/petewarden/tensorflow_makefile/blob/master/tensorflow/models/image/alexnet/alexnet_benchmark.py). Most likely, this file will be changed to more efficient and correct implementation. This file is called by different benchmarking scripts and in general there is no need to call it manually.

For example, the following commands
```
python time_tf.py --model=alexnet.kaffe.py --name=AlexNet --shape='128,227,227,3' --num-warmup-iters=2 --num-iters=10
```
benchmarks TensorFlow stored in `alexnet.kaffe.py` file. The model class name is 'AlexNet'. The batch size is 128. Model is benchmarked for 10 iterations (batches) with 2 warmup iterations (to do all memory allocations and initializations). 
