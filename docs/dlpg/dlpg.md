# Deep Learning Performance Guide
## Terminology
<p align="center"><img src="https://raw.githubusercontent.com/HewlettPackard/dlcookbook-dlbs/master/docs/dlpg/imgs/terminology.png"/></p>

1. `Neural network`. Composite differentiable function `y=f(x)` that learns to compute
   a desired output `y` given a particular input `x`.

2. `Loss function` (`cost function`) is a function `loss(predictions, true_outputs)` that computes difference between neural network's predictions `predictions` and true outputs `true_outputs`. When predictions are close to true outputs, loss function outputs small values. The goal of training a neural network is to minimize a loss function.

3. `Optimizer` (`optimization algorithm`, `training algorithm`) is a mathematical algorithm that adjusts neural network's parameters given values provided by a loss function. Usually, it is some form of stochastic gradient descent (SGD) like SGD, SGD with momentum, Adam, AdaGrad, RMSProp etc.

4. `Dataset` or `training data`. A collection of know data samples that is used to optimize neural networks. Dataset is called `unlabeled` if it contains only `x`s i.e. we do not know the desired output. Dataset is called `labeled` if every data sample is a pair `(x, y)` that tells what the desired output `y` is given input `x`.

5. `Epoch` One pass over `training data`. Neural network optimization process is sequential in its nature. Optimization algorithm takes small subset of `training data` and makes a small update to neural networks' parameters to better fit the data. Then, it takes another small subset and makes another update. There can be millions of updates during one training process. When NN has used all data samples in a `training dataset`, it is called one `epoch`. Once it is done, optimization algorythm randomly shuffles data and starts this process all over again. There can be required many `epochs` to train a neural network.

6. `Batch` A small subset of `training data` used by `optimizer` to perform one update. Usually a `batch` contains several hundreds of data samples. However, researchers have developed a specialized training algorithms that can use up to 32 thousand data samples in one `batch`. Why is it a problem? In short, large `batches` result in worse model accuracy.

7. `Iteration` is basically the synonym for one `batch`. When NN processes one `batch`, it is called it has done one `iteration`. One training `iteration` is usually composed of the following steps: (1) Prepare batch, (2) run forward pass and compute NN outputs, (3) take actual and desired outputs and compute an error, (4) use that error to compute gradients and updates, (5) optionally average updates across multiple GPUs and (6) apply updates. One `inference` iteration is usually composed of two steps: (1) prepare and ingest data into GPU memory and (2) run forward pass and compute NN outputs.

8. `Predictions` Actual outputs of a neural network.

9. `Errors` A signal that tells how close predictions are to the desired outputs. It is usually a vector of numbers.

10. `True labels` The desired output.

11. `Worker` One entity participating in a distributed training. The notion of a `worker` may be used to refer to different entities depending on a context. Usually, it is either a __single GPU__ or a __single multi-GPU node__.

12. `Effective batch` Also known as `global batch` or `algorithmic batch` or just `batch`. Total data samples used to update NN parameters in each iteration. It is computed as per-device batch size multiplied by number of compute devices. For instance, if 8 GPUs are used to train a single NN and each GPU uses 128 images in one iteration, `effective batch` is computed to be 128*8=1024 images. The value of this parameter affects convergence rate and final accuracy, that's why it is called `algorithmic batch`.

13. `Replica batch` Batch used by each individual worker. In the example above (#12) , `replica batch` is 128 data samples (images) per GPU.

14. `Strong scaling` is a way to distribute a work given presence of more than one compute devices, for instance, GPUs. In `strong scaling`, size of input problem remains the same no matter how many compute devices we use. The more compute devices we use, the less work is assigned to each compute device. Imagine a `batch` of 1024 images. With one worker, it gets all 1024 images. With 2 workers each gets 512 images. With 8 workers each gets 128 images and so on. More information on `strong scaling` is [here](https://developer.hpe.com/blog/scaling-deep-learning-workloads).

15. `Weak scaling` is a way yo distribute a work given presence of more than one compute devices like GPUs. In `weak scaling`, size of input per compute device remains constant no matter how many compute devices (e.g. GPUs) we use. Adding more compute devices does not change amount of work each device is assigned with per iteration but increases `effective batch`. For instance, assume one worker uses `batch` of 128 images. With two workers each worker gets the same 128 images increasing `effective batch` to 256 images. More information on `weak scaling` is [here](https://developer.hpe.com/blog/scaling-deep-learning-workloads).

16. `Synthetic data` Training and inference requires input data. For instance, during training process NN uses `training data` to optimize its parameters. Getting data into GPU memory is called an `ingestion` process and may be computationally expensive. For instance, we want to transform input images on the fly. We usually do that on CPUs - while NN processes current `batch`, CPU is busy with preparing next `batch` to make GPU busy all the time. Sometimes, due to different reasons, like large `batches` or expensive data transformations, CPUs do not keep up with GPUs what slows down the process. During benchmarks, we sometimes want to benchmark either GPU compute capabilities or GPU <-> GPU interconnect. In order to be able to do it, we must ensure that ingestion pipeline is not a bottleneck, sort of disable it. This is where term `synthetic data` comes into play. `Synthetic data` means random data of appropriate size in host or GPU memory that makes ingestion pipeline almost invisible in terms of latencies/computation requirements.


## Deep Learning Performance Guide
### Chart Axes
First step to explore data that DLPG has is to specify chart's X- and Y- axes.
<p align="center"><img src="https://raw.githubusercontent.com/HewlettPackard/dlcookbook-dlbs/master/docs/dlpg/imgs/axes.png"/></p>

##### Vertical Axis
<p align="center"><img src="https://raw.githubusercontent.com/HewlettPackard/dlcookbook-dlbs/master/docs/dlpg/imgs/axis_y.png"/></p>
Vertical, or Y-axis, corresponds to target variable we are interested in. Think about it as comparison criteria. Currently, DLPG provides two options:

1. `Batch time (ms)` An average time in milliseconds of one `iteration` (time to process one batch). Two things to keep in mind:
   - The smaller this time is the better. So, think about __minimizing__ this variable.
   - Depending on a phase (`training` or `inferencing`), the `batch time` describes different steps. Read about `iteration` term.

2. `Throughput` An average number of data samples NN processes in one seconds (samples/second). For instance, major of convolutional NNs defines this to be images/second. Several things to keep in mind:
   - Think about it as `effective throughput`. This value defines aggregated throughput for a configuration. So, if 8 GPUs were used, this variable defines aggregated throughput of all 8 GPUs.
   - The bigger this number is the better. So, think about __maximizing__ this variable.

##### Horizontal Axis
<p align="center"><img src="https://raw.githubusercontent.com/HewlettPackard/dlcookbook-dlbs/master/docs/dlpg/imgs/axis_x.png"/></p>
Horizontal, or X-axis, specifies what aspect of benchmarks we want to compare. Currently, we have the following variables:

1. `Data Type` Specifies what and where `training dataset` is located. It can be a `synthetic data` or some real input dataset (for instance, a bunch of images). Possible values of this variable are:
   - `Real (DRAM)` Input dataset is located in host memory.
   - `Real (Weka IO)` Input dataset is hosted by high performance WEKA.IO storage.
   - `Real (local NVMe)` Input dataset is located on local NVMe drive.
   - `Real (local SSD)` Input dataset is located on local SSD drive or some sort of RAID.
   - `Synthetic (DRAM)` Read about `Synthetic data` term. Basically, means there's no dataset and overhead associated with ingesting data into GPU memory. Benchmarks with `synthetic data` demonstrates highest compute capabilities and serves as a target performance for workloads with real data. This means that we can compare implementations of ingestion pipelines / storage offerings / various network interconnects by how close in terms of performance they are to benchmarks with `synthetic data`.

2. `Effective batch size` Read about `Effective batch` term. Total number of data samples in one `batch` processed in one `iteration`.

3. `Framework`. A Deep Learning or any other software framework.

4. `Model` A neural network model. List of neural network models along with brief description and visualization is located [here](https://hewlettpackard.github.io/dlcookbook-dlbs/#/models/models?id=models).

5. `Num processing units` Number of compute devices used in benchmarks. In most cases, this is just number of GPUs. For instance, 8 means that 8 GPUs were simultaneously used to train one Neural Network. __You should use variable to plot weak/strong scaling reports.__

6. `Phase` Either `training` or `inference`. Training means building a model. Inference means using that model in production.

7. `Precision` Either `single (FP32)`, `half (FP16)` or `INT8`. Defines type used by NN to store its parameters. Single precision (FP32) is 32-bit (4 bytes) floating point format. It is supported by almost all compute devices. The most studied case and NNs that use this data type converge quite well to a good solution. Half precision (FP16) uses 16-bit (2-bytes) floating point representation. Takes less memory and much faster on some of the GPUs, like NVIDIA GPUS, that can use tensor cores with half precision to significantly increase the compute capability. Less studied, but very active research area. Not all models can be trained to a good solution with half precision, but researchers are actively working on this. The INT8 format uses 8-bit (1 byte) representation. Nowadays, is mostly used for inference on such devices as NVIDIA P4 that natively support INT8 operations.

8. `Processing unit` Main compute device for training/inference. Values range from CPUs to GPUs models. If CPUs are selected, it means that training was done on CPU.

9. `Replica Batch Size` Read about `replica batch` term. Usually, a per-GPU size of a `batch`.

10. `Server` Model of a server. In some of the titles, `1:1` and `1:2` means CPU to GPU ratio. The `1:1` means a two CPU / two GPU server where each GPU is connected via PCIe lanes to its own CPU. Such configurations are usually calle High Performance Compute configurations. The `2:1` means a two CPU / two GPU system where all 2 GPUs are connected via PCIe lanes to one of the CPU. Such configurations are usually called Deep Learning configurations.

11. `Software` A brief description of a benchmark software. Includes information on framework and its version, benchmark backend and runtime - docker or singularity container or baremetal.

### Filtering benchmarks
Once axes have been selected, DLPG opens `filter` dialog where users narrow down what configurations they are interested in.

<p align="center"><img src="https://raw.githubusercontent.com/HewlettPackard/dlcookbook-dlbs/master/docs/dlpg/imgs/filters.png"/></p>

Most of the filters on this page are the same as variables defined for horizontal axis described above: `Phase`, `Precision`, `Data` (`Data Type`), `Server`, `Processing unit`, `Count` (`Num processing units`), `Framework`. One comment here is that the `Count` filter corresponds to `Processing unit` within one node e.g. number of selected compute devices inside one node. Other filters are the following:

1. `Cluster size` and `Interconnect`. Define number of `Server` and network interconnect between them. Currently, we have only a small number of benchmarks that ran with 2 nodes.
2. `Scaling`. Read about `weak scaling`/`strong scaling` terms. In this particular context, is basically used to select benchmarks based on effective batch size. In another words, this is a qualifier for `Batch size` filter.
   - When user selects `weak`, the `Batch size` is considered to be a `replica batch`.
   - When user selects `strong`, the `Batch size` is considered to be an `effective batch`.
3. `Batch size` - either `replica batch` or `effectve batch` depending on `Scaling` filter.

The `Aggregation method` filter provides option on how to aggregate multiple results. Based on user selection and/or data in a database, for every point on a chart we can have multiple benchmarks (for instance, multiple trials of the same configuration, or, user did not specify some of the filters). DLPG supports three types of aggregation methods - `mix`, `max` and `average`. I would not recommend using `average` for now. With `Batch time` y-axis variable, users should select `min` for best performance, and for `Throughput` y-axis variable users should select `max` for best performance.

The `Group by` is a special filter that allows generating multiple charts in one step. Think about it as a secondary dimension we build chart for (the first one is a y-axis variable).

> For instance, assume user is interested in comparing throughput (images/sec) for various NN models and various precision values. On the first page user has selected a `Model` as a horizontal variable with 5 values - AlexNet, GoogleNet, InceptionV3, ResNet50 and VGG16. On this page user selects `Group by` filter for `Precision` and selects `FP32` and `FP16` as values. This makes it possible to compare throughput of these neural network models for single/half precision parameter store.

If `Group by` is not selected, users must specify series name `Name of the series`.
