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
2. ``TensorFlow``with [nvcnn benchmarks](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/python/nvcnn_benchmarks) backend. Models are implemented in a single [file](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/nvcnn_benchmarks/nvcnn.py).
3. ``TensorFlow`` with [nvtfcnn benchmark](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/python/nvcnn_benchmarks) backend. Currently, DLBS runs files distributed with specific containers from NGC.
3. ``Caffe2`` with [caffe2 benchmarks](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/python/caffe2_benchmarks)
   backend. Models are defined in [models](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/python/caffe2_benchmarks/models)
   folder.
4. ``Caffe`` (Intel/BVLC/NVIDIA) models are defined as inference and training prototxt files in
   [models](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/models) folder. This folder contains
   multiple subfolders - one subfolder for one model.
   **Important**: The inference and training descriptors are not true Caffe descriptors. They cannot be used
   directly by Caffe tool. A Caffe [luncher](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/scripts/launchers/caffe.sh)
   does some preprocessing, like setting batch size, selecting appropriate data source and specifying
   type of data if supported (NVIDIA fork currently).
5. ``MXNet`` with [mxnet benchmarks](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/python/mxnet_benchmarks)
   backend. Models are defined in [models](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/python/mxnet_benchmarks/models)
   folder.
6. ``TensorRT`` uses Caffe\`s prototxt inference models defined in [models](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/models)
   folder.
7. ``PyTorch`` with [pytorch benchmarks](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/python/pytorch_benchmarks) backend. Models are defined in [models](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/python/pytorch_benchmarks/models) folder.

## Supported models

The following table summarizes models we currently support. We try to make sure that every
model implementation is the same across all frameworks. There may still be differences - I am working
on it now. The source code for the models contains references of the original implementations.

> Pay attention to shapes of input images - they are different for different models. This
> introduces different requirements to minimal spatial dimensions of images in your dataset.

<table>
  <tr>
    <th>model</th><th>Name</th><th>Input shape*</th><th>#Parameters</th><th>#Model size (Mb)</th><th>TensorFlow</th><th>Caffe</th><th>TensorRT</th><th>Caffe2</th><th>MXNet</th><th>PyTorch</th>
  </tr>
  <tr>
    <td>acoustic_model</td><td><a href="http://ethereon.github.io/netscope/#/gist/10f5dee56b6f7bbb5da26749bd37ae16">AcousticModel</a></td>
    <td>540</td>
    <td>34,678,784</td><td>133</td>
    <td><a href="https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/tf_cnn_benchmarks/models/acoustic_model.py">Impl</a></td>
    <td colspan="2"><a href="https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/models/acoustic_model">Impl</a></td>
    <td><a href="https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/caffe2_benchmarks/models/acoustic_model.py">Impl</a></td>
    <td><a href="https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/mxnet_benchmarks/models/acoustic_model.py">Impl</a></td>
    <td><a href="https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/pytorch_benchmarks/models/acoustic_model.py">Impl</a></td>
  </tr>
  <tr>
    <td>alexnet</td><td><a href="http://ethereon.github.io/netscope/#/gist/5c94a074f4e4ac4b81ee28a796e04b5d">AlexNet</a></td>
    <td>3x227x227</td>
    <td>62,378,344</td><td>238</td>
    <td><a href="https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/tf_cnn_benchmarks/models/alexnet_model.py">Impl</a></td>
    <td colspan="2"><a href="https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/models/alexnet_owt">Impl</a></td>
    <td><a href="https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/caffe2_benchmarks/models/alexnet_owt.py">Impl</a></td>
    <td><a href="https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/mxnet_benchmarks/models/alexnet_owt.py">Impl</a></td>
    <td><a href="https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/pytorch_benchmarks/models/alexnet_owt.py">Impl</a></td>
  </tr>
  <tr>
    <td>alexnet_owt</td><td><a href="http://ethereon.github.io/netscope/#/gist/dc85cc15d59d720c8a18c4776abc9fd5">AlexNetOWT</a></td>
    <td>3x227x227</td>
    <td>61,100,840</td><td>233</td>
    <td><a href="https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/tf_cnn_benchmarks/models/alexnet_model.py">Impl</a></td>
    <td colspan="2"><a href="https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/models/bvlc_alexnet">Impl</a></td>
    <td><a href="https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/caffe2_benchmarks/models/alexnet.py">Impl</a></td>
    <td><a href="https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/mxnet_benchmarks/models/alexnet.py">Impl</a></td>
    <td><a href="https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/pytorch_benchmarks/models/alexnet.py">Impl</a></td>
  </tr>
  <tr>
    <td>deep_mnist</td><td><a href="http://ethereon.github.io/netscope/#/gist/9c75cd95891207082bd42264eb7a2706">DeepMNIST</a></td>
    <td>1x28x28</td>
    <td>11,972,510</td><td>46</td>
    <td><a href="https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/tf_cnn_benchmarks/models/deepmnist_model.py">Impl</a></td>
    <td colspan="2"><a href="https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/models/deep_mnist">Impl</a></td>
    <td><a href="https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/caffe2_benchmarks/models/deep_mnist.py">Impl</a></td>
    <td><a href="https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/mxnet_benchmarks/models/deep_mnist.py">Impl</a></td>
    <td><a href="https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/pytorch_benchmarks/models/deep_mnist.py">Impl</a></td>
  </tr>
  <tr>
    <td>googlenet</td><td><a href="http://ethereon.github.io/netscope/#/gist/4325909f3683e51eaf93fdaeed6b2a9b">GoogleNet</a></td>
    <td>3x224x224</td>
    <td>6,998,552</td><td>27</td>
    <td><a href="https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/tf_cnn_benchmarks/models/googlenet_model.py">Impl</a></td>
    <td colspan="2"><a href="https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/models/bvlc_googlenet">Impl</a></td>
    <td><a href="https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/caffe2_benchmarks/models/googlenet.py">Impl</a></td>
    <td><a href="https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/mxnet_benchmarks/models/googlenet.py">Impl</a></td>
    <td><a href="https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/pytorch_benchmarks/models/googlenet.py">Impl</a></td>
  </tr>
  <tr>
    <td>inception3</td><td><a href="http://ethereon.github.io/netscope/#/gist/04a797f778a7d513a9b52af4c1dbee4e">Inception3</a></td>
    <td rowspan=2>3x299x299</td>
    <td>23,869,094</td><td>91</td>
    <td rowspan="2"><a href="https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/tf_cnn_benchmarks/models/inception_model.py">Impl</a></td>
    <td colspan=2><a href="https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/models/inception3">Impl</a></td>
    <td rowspan=2><a href="https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/caffe2_benchmarks/models/inception.py">Impl</a></td>
    <td rowspan=2><a href="https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/mxnet_benchmarks/models/inception.py">Impl</a></td>
    <td rowspan=2><a href="https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/pytorch_benchmarks/models/inception.py">Impl</a></td>
  </tr>
  <tr>
    <td>inception4</td><td><a href="http://ethereon.github.io/netscope/#/gist/8fdab7a3ea5bceb9169832dfd73b5e31">Inception4</a></td>
    <td>42,743,133</td><td>163</td>
    <td colspan=2><a href="https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/models/inception4">Impl</a></td>
  </tr>
  <tr>
    <td>overfeat</td><td><a href="http://ethereon.github.io/netscope/#/gist/ebfeff824393bcd66a9ceb851d8e5bde">Overfeat</a></td>
    <td>3x231x231</td>
    <td>145,920,872</td><td>557</td>
    <td><a href="https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/tf_cnn_benchmarks/models/overfeat_model.py">Impl</a></td>
    <td colspan=2><a href="https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/models/overfeat">Impl</a></td>
    <td><a href="https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/caffe2_benchmarks/models/overfeat.py">Impl</a></td>
    <td><a href="https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/mxnet_benchmarks/models/overfeat.py">Impl</a></td>
    <td><a href="https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/pytorch_benchmarks/models/overfeat.py">Impl</a></td>
  </tr>
  <tr>
    <td>resnet18</td><td><a href="http://ethereon.github.io/netscope/#/gist/649e0fb6c96c60c9f0abaa339da3cd27">ResNet18</a></td>
    <td rowspan=7>3x224x224</td>
    <td>11,703,485</td><td>45</td>
    <td rowspan="7"><a href="https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/tf_cnn_benchmarks/models/resnet_model.py">Impl</a></td>
    <td colspan=2><a href="https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/models/resnet18">Impl</a></td>
    <td rowspan="7"><a href="https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/caffe2_benchmarks/models/resnet.py">Impl</a></td>
    <td rowspan="7"><a href="https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/mxnet_benchmarks/models/resnet.py">Impl</a></td>
    <td rowspan="7"><a href="https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/pytorch_benchmarks/models/resnet.py">Impl</a></td>
  </tr>
  <tr>
    <td>resnet34</td><td><a href="http://ethereon.github.io/netscope/#/gist/277a9604370076d8eed03e9e44e23d53">ResNet34</a></td>
    <td>21,819,085</td><td>84</td>
    <td colspan=2><a href="https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/models/resnet34">Impl</a></td>
  </tr>
  <tr>
    <td>resnet50</td><td><a href="http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006">ResNet50</a></td>
    <td>25,610,269</td><td>98</td>
    <td colspan="2"><a href="https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/models/resnet50">Impl</a></td>
  </tr>
  <tr>
    <td>resnet101</td><td><a href="http://ethereon.github.io/netscope/#/gist/b21e2aae116dc1ac7b50">ResNet101</a></td>
    <td>44,654,608</td><td>171</td>
    <td colspan="2"><a href="https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/models/resnet101">Impl</a></td>
  </tr>
  <tr>
    <td>resnet152</td><td><a href="http://ethereon.github.io/netscope/#/gist/d38f3e6091952b45198b">ResNet152</a></td>
    <td>60,344,387</td><td>231</td>
    <td colspan="2"><a href="https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/models/resnet152">Impl</a></td>
  </tr>
  <tr>
    <td>resnet200</td><td><a href="http://ethereon.github.io/netscope/#/gist/38a20d8dd1a4725d12659c8e313ab2c7">ResNet200</a></td>
    <td>64,850,035</td><td>248</td>
    <td colspan=2><a href="https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/models/resnet200">Impl</a></td>
  </tr>
  <tr>
    <td>resnet269</td><td><a href="http://ethereon.github.io/netscope/#/gist/fbf7c67565523a9ac2c349aa89c5e78d">ResNet269</a></td>
    <td>102,326,456</td><td>391</td>
    <td colspan=2><a href="https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/models/resnet269">Impl</a></td>
  </tr>
  <tr>
    <td>vgg11</td><td><a href="http://ethereon.github.io/netscope/#/gist/5550b93fb51ab63d520af5be555d691f">VGG11</a></td>
    <td rowspan=4>3x224x224</td>
    <td>132,863,336</td><td>507</td>
    <td rowspan="4"><a href="https://github.hpe.com/labs/dlcookbook/blob/master/python/tf_cnn_benchmarks/models/vgg_model.py">Impl</a></td>
    <td colspan="2"><a href="https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/models/vgg11">Impl</a></td>
    <td rowspan="4"><a href="https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/caffe2_benchmarks/models/vgg.py">Impl</a></td>
    <td rowspan="4"><a href="https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/mxnet_benchmarks/models/vgg.py">Impl</a></td>
    <td rowspan="4"><a href="https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/pytorch_benchmarks/models/vgg.py">Impl</a></td>
  </tr>
  <tr>
    <td>vgg13</td><td><a href="http://ethereon.github.io/netscope/#/gist/a96ba317064a61b22a1742bd05c54816">VGG13</a></td>
    <td>133,047,848</td><td>508</td>
    <td colspan="2"><a href="https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/models/vgg13">Impl</a></td>
  </tr>
  <tr>
    <td>vgg16</td><td><a href="http://ethereon.github.io/netscope/#/gist/050efcbb3f041bfc2a392381d0aac671">VGG16</a></td>
    <td>138,357,544</td><td>528</td>
    <td colspan="2"><a href="https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/models/vgg16">Impl</a></td>
  </tr>
  <tr>
    <td>vgg19</td><td><a href="http://ethereon.github.io/netscope/#/gist/f9e55d5947ac0043973b32b7ff51b778">VGG19</a></td>
    <td>143,667,240</td><td>548</td>
    <td colspan="2"><a href="https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/models/vgg19">Impl</a></td>
  </tr>
  <tr>
    <td>inception_resnet_v2</td><td>InceptionResNetV2</td>
    <td>3x299x299</td>
    <td>NA</td><td>NA</td>
    <td>NA</td>
    <td colspan="2">NA</td>
    <td><a href="https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/caffe2_benchmarks/models/inception_resnet_v2.py">Impl</a></td>
    <td><a href="https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/mxnet_benchmarks/models/inception_resnet_v2.py">Impl</a></td>
    <td>NA</td>
  </tr>
  <tr>
    <td>deep_speech2</td><td>DeepSpeech2</td>
    <td>1x200x161</td>
    <td>NA</td><td>NA</td>
    <td>NA</td>
    <td colspan="2">NA</td>
    <td>NA</td>
    <td><a href="https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/mxnet_benchmarks/models/deep_speech2.py">Impl</a></td>
    <td>NA</td>
  </tr>
</table>

`*` Excluding batch dimension. The dimensions of input shape depend on a model architecture. For convolution neural networks it is [C,H,W] (channels, height, width). For neural nets accepting vectors it is [N] where N is the number of features. DeepSpeech2's input shape is [1,L,N] where L is the length on an input sequence and N is the number of features per time step.

The experimenter script accepts ``--model`` command line argument that specifies model to benchmark.

1. __AlexNet__ Same as [BVLC Caffe's version](https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet) _without_ grouped convolutions in layers 2, 4 and 5 (`group=1`). This does not significantly change number of  trainable parameters but does change computational profile - roughly from 0.7 gFLOP to 1.14 gFLOP for forward pass.
2. __DeepMNIST__ A fully-connected architecture mentioned [here](http://yann.lecun.com/exdb/mnist/) described in this [paper](http://arxiv.org/abs/1003.0358).
3. __AcousticModel__ A fully-connected architecture that's typically used in hybrid HMM-DNN speech recognition systems (English language) for acoustic modeling. Similar to a speech network described in Large Scale Distributed Deep Networks [paper](https://research.google.com/archive/large_deep_networks_nips2012.html).
4. __GoogleNet__ Same as version implemented in BVLC Caffe [here](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet). Reference publication is [here](http://arxiv.org/abs/1409.4842).
5. __Inception3__ and __Inception4__ are based on original implementation in [tf_cnn_benchmarks](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/tf_cnn_benchmarks/inception_model.py). Inception3 model publication is [here](http://arxiv.org/abs/1512.00567). Inception4 publication is [here](http://arxiv.org/abs/1602.07261).
6. __Overfeat__ A model described in this [paper](https://arxiv.org/pdf/1312.6229.pdf). Based on Google's tf_cnn_benchmarks with additional dropout operators applied to 6th and 7th layers as described in the paper.
7. __ResNets__ are the reference implementations. ResNet18 and ResNet34 are based on this [implementation](https://github.com/antingshen/resnet-protofiles).
ResNet50, 101 and 152 are based on this [implementation](https://github.com/KaimingHe/deep-residual-networks). ResNet200 and 269 are based on ResNet152's descriptor. I re-implemented ResNets in tf_cnn_benchmarks. See details [here](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/tf_cnn_benchmarks/resnet_model.py).
8. __VGGs__ VGG16 and VGG19 models are taken from [here](https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md) and [here](https://gist.github.com/ksimonyan/3785162f95cd2d5fee77#file-readme-md). They are improved versions of architectures described in this [paper](http://arxiv.org/pdf/1409.1556). VGG11 and VGG13 descriptors are based on VGG16.
9. __DeepSpeech2__ is a proxy implementation similar to [PaddlePaddle](https://github.com/PaddlePaddle/DeepSpeech) implementation without actual input data. It can be used to estimate performance of models of this family. Some implementation details are presented [here](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/python/mxnet_benchmarks/models/deep_speech2.py#L15). 
