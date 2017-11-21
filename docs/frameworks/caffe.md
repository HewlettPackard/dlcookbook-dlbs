# __Caffe__

Various forks of Caffe are integrated into deep learning benchmarking suite. This integration is pretty straightforward. We use standard caffe's time and train command to benchmark training/inference. A launcher shell script, [Caffe wrapper](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/scripts/launchers/caffe.sh), exists that serves as mediator taking experiment parameters and invoking caffe.

## Standalone run
As is, there is no special tool to run caffe from command line. Use caffe directly passing all parameters that it accepts.

## Models
For Caffe's forks, all models are stored in [models](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/models) directory in the root folder of the project.
Complete list of supported models can be found on the [model](/models/models.md?id=supported-models) page.

## Adding new model
To add new model, several steps need to be performed:
1. Create new folder with a model id name in [models](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/models).
2. Create there two files. One is a caffe training prototxt file ending with `.training.prototxt`. The other one is a Caffe deployment (inference) prototxt descriptor ending with `.inference.prototxt`.
3. Standard Caffe training descriptor must be modified in order to (1) support synthetic and real data, (2) support configurable batch size and (3) support float16/float32 inference in NVIDIA Caffe:
    1. Supporting synthetic/real data with configurable batch size. This is an example of input data specification that enables this functionality:

       ```bash
       #synthetic layer {
       #synthetic   name: "data"
       #synthetic   type: "Input"
       #synthetic   top: "data"
       #synthetic   input_param { shape: { dim: __EXP_DEVICE_BATCH__ dim: 3 dim: 227 dim: 227 } }
       #synthetic }
       #synthetic layer {
       #synthetic   name: "label"
       #synthetic   type: "Input"
       #synthetic   top: "label"
       #synthetic   input_param { shape: { dim: __EXP_DEVICE_BATCH__ dim: 1 } }
       #synthetic }

       #data layer {
       #data   name: "data"
       #data   type: "Data"
       #data   top: "data"
       #data   top: "label"
       #data   include {
       #data     phase: TRAIN
       #data   }
       #data   transform_param {
       #data     mirror: __CAFFE_MIRROR__
       #data     crop_size: 227
       #data     mean_file: "__CAFFE_DATA_MEAN_FILE__"
       #data   }
       #data   data_param {
       #data     source: "__CAFFE_DATA_DIR__"
       #data     batch_size: __EXP_DEVICE_BATCH__
       #data     backend: __CAFFE_DATA_BACKEND__
       #data   }
       #data }
       ```
    2. Supporting float16/float32 types for NVIDIA Caffe. Copy-paste these lines in the beginning of the training descriptor:

       ```bash
       #precision default_forward_type: __FORWARD_TYPE___
       #precision default_backward_type: __BACKWARD_TYPE___
       #precision default_forward_math: __FORWARD_MATH___
       #precision default_backward_math: __BACKWARD_MATH___
       ```
For simple examples, study deep MNIST [deployment](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/models/deep_mnist/deep_mnist.inference.prototxt) and [training](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/models/deep_mnist/deep_mnist.training.prototxt) descriptors.
