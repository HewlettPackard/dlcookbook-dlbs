# __Models__

The original idea I had in mind was to have all models in one format (Caffe's prototxt)
and then convert those models into other formats. For instance, Caffe to TensorFlow converter
exists. Caffe2 also provides a converter from legacy Caffe format.

It turned out that this was not the best option. For instance, with Caffe2's converter
I was getting worse performance comparing to native Caffe2 python model implementation.

So, I decided to go with framework specific implementations. Recently, Open Neural Network
Exchange format ([ONNX](https://github.com/onnx/onnx)) was announced.
This may become a solution in future releases.

In current version, models are defined depending on which framework backend is used:

1. ``TensorFlow`` with [tf_cnn_benchmarks](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/python/tf_cnn_benchmarks)
   backend. Models are defined in python code in this [folder](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/python/tf_cnn_benchmarks)
   in python files ending with **_model.py**.
2. ``Caffe2`` with [caffe2 benchmarks](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/python/caffe2_benchmarks)
   backend. Models are defined in [models](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/python/caffe2_benchmarks/models)
   folder.
3. ``Caffe`` (Intel/BVLC/NVIDIA) models are defined as inference and training prototxt files in
   [models](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/models) folder. This folder contains
   multiple subfolders - one subfolder for one model.
   **Important**: The inference and training descriptors are not true Caffe descriptors. They cannot be used
   directly by Caffe tool. A Caffe [luncher](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/scripts/launchers/caffe.sh)
   does some preprocessing, like setting batch size, selecting appropriate data source and specifying
   type of data if supported (NVIDIA fork currently).
4. ``MXNet`` with [mxnet benchmarks](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/python/mxnet_benchmarks)
   backend. Models are defined in [models](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/python/mxnet_benchmarks/models)
   folder.
5. ``TensorRT`` uses Caffe\`s prototxt inference models defined in [models](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/models)
   folder.

## Supported models

The following table summarizes models we currently support. We try to make sure that every
model implementation is the same across all frameworks. There may still be differences - I am working
on it now. The source code for the models contains references of the original implementations.

<table>
  <tr>
    <th>model</th><th>Name</th><th>#Parameters</th><th>#Model size (Mb)</th><th>TensorFlow</th><th>Caffe</th><th>TensorRT</th><th>Caffe2</th><th>MXNet</th>
  </tr>
  <tr>
    <td>alexnet</td><td>[AlexNet](http://ethereon.github.io/netscope/#/gist/f2e4825a8d4f8a3609cefd7ffadc910a)</td>
    <td>60,965,224</td><td>233</td>
    <td>[Impl](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/tf_cnn_benchmarks/alexnet_model.py)</td>
    <td colspan="2">[Impl](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/models/bvlc_alexnet)</td>
    <td>[Impl](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/caffe2_benchmarks/models/alexnet.py)</td>
    <td>[Impl](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/mxnet_benchmarks/models/alexnet.py)</td>
  </tr>
  <tr>
    <td>deep_mnist</td><td>[DeepMNIST](http://ethereon.github.io/netscope/#/gist/9c75cd95891207082bd42264eb7a2706)</td>
    <td></td><td></td>
    <td>[Impl](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/tf_cnn_benchmarks/deepmnist_model.py)</td>
    <td colspan="2">[Impl](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/models/deep_mnist)</td>
    <td>[Impl](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/caffe2_benchmarks/models/deep_mnist.py)</td>
    <td>[Impl](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/mxnet_benchmarks/models/deep_mnist.py)</td>
  </tr>
  <tr>
    <td>eng_acoustic_model</td><td>[EngAcousticModel]()</td>
    <td></td><td></td>
    <td>[Impl](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/tf_cnn_benchmarks/engacoustic_model.py)</td>
    <td colspan="2">[Impl](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/models/eng_acoustic_model)</td>
    <td>[Impl](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/caffe2_benchmarks/models/eng_acoustic_model.py)</td>
    <td>[Impl](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/mxnet_benchmarks/models/eng_acoustic_model.py)</td>
  </tr>
  <tr>
    <td>googlenet</td><td>[GoogleNet](http://ethereon.github.io/netscope/#/gist/4325909f3683e51eaf93fdaeed6b2a9b)</td>
    <td>6,998,552</td><td>27</td>
    <td>[Impl](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/tf_cnn_benchmarks/googlenet_model.py)</td>
    <td colspan="2">[Impl](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/models/bvlc_googlenet)</td>
    <td>[Impl](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/caffe2_benchmarks/models/googlenet.py)</td>
    <td>[Impl](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/mxnet_benchmarks/models/googlenet.py)</td>
  </tr>
  <tr>
    <td>inception3</td><td>Inception3</td>
    <td></td><td></td>
    <td rowspan="2">[Impl](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/tf_cnn_benchmarks/inception_model.py)</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>inception4</td><td>Inception4</td>
    <td></td><td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>overfeat</td><td>Overfeat</td>
    <td></td><td></td>
    <td>[Impl](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/tf_cnn_benchmarks/overfeat_model.py)</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>resnet18</td><td>ResNet18</td>
    <td></td><td></td>
    <td framework="tensorflow"></td>
    <td framework="caffe"></td>
    <td framework="tensorrt"></td>
    <td rowspan="7" framework="caffe2">[Impl](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/caffe2_benchmarks/models/resnet.py)</td>
    <td rowspan="7" framework="mxnet">[Impl](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/mxnet_benchmarks/models/resnet.py)</td>
  </tr>
  <tr>
    <td>resnet34</td><td>ResNet34</td>
    <td></td><td></td>
    <td framework="tensorflow"></td>
    <td framework="caffe"></td>
    <td framework="tensorrt"></td>
  </tr>
  <tr>
    <td>resnet50</td><td>[ResNet50](http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006)</td>
    <td>25,610,269</td><td>98</td>
    <td rowspan="3">[Impl](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/tf_cnn_benchmarks/resnet_model.py)</td>
    <td colspan="2">[Impl](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/models/resnet50)</td>
  </tr>
  <tr>
    <td>resnet101</td><td>[ResNet101](http://ethereon.github.io/netscope/#/gist/b21e2aae116dc1ac7b50)</td>
    <td>44,654,608</td><td>171</td>
    <td colspan="2">[Impl](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/models/resnet101)</td>
  </tr>
  <tr>
    <td>resnet152</td><td>[ResNet152](http://ethereon.github.io/netscope/#/gist/d38f3e6091952b45198b)</td>
    <td>60,344,387</td><td>231</td>
    <td colspan="2">[Impl](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/models/resnet152)</td>
  </tr>
  <tr>
    <td>resnet200</td><td>ResNet200</td>
    <td></td><td></td>
    <td framework="tensorflow"></td>
    <td framework="caffe"></td>
    <td framework="tensorrt"></td>
  </tr>
  <tr>
    <td>resnet269</td><td>ResNet269</td>
    <td></td><td></td>
    <td framework="tensorflow"></td>
    <td framework="caffe"></td>
    <td framework="tensorrt"></td>
  </tr>
  <tr>
    <td>vgg11</td><td>VGG11</td>
    <td></td><td></td>
    <td>[Impl](https://github.hpe.com/labs/dlcookbook/blob/master/python/tf_cnn_benchmarks/vgg_model.py)</td>
    <td colspan="2"></td>
    <td rowspan="4">[Impl](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/caffe2_benchmarks/models/vgg.py)</td>
    <td rowspan="4">[Impl](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/mxnet_benchmarks/models/vgg.py)</td>
  </tr>
  <tr>
    <td>vgg13</td><td>VGG13</td>
    <td></td><td></td>
    <td></td>
    <td colspan="2"></td>
  </tr>
  <tr>
    <td>vgg16</td><td>[VGG16](http://ethereon.github.io/netscope/#/gist/050efcbb3f041bfc2a392381d0aac671)</td>
    <td>138,357,544</td><td>528</td>
    <td rowspan="2">[Impl](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/tf_cnn_benchmarks/vgg_model.py)</td>
    <td colspan="2">[Impl](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/models/vgg16)</td>
  </tr>
  <tr>
    <td>vgg19</td><td>[VGG19]()</td>
    <td>143,667,240</td><td>548</td>
    <td colspan="2">[Impl](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/models/vgg19)</td>
  </tr>
</table>

The experimenter script accepts ``--model`` command line argument that specifies model to benchmark.
