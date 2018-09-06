
# Convolutional neural network training scripts

These scripts implement a number of popular CNN models and demonstrate efficient
single-node training on multi-GPU systems. They can be used for benchmarking, or
as a starting point for implementing and training your own network.

Common utilities for defining CNN networks and performing basic training are
located in the nvutils directory. Use of nvutils is demonstrated in the model
scripts.

For parallelization, we use the Horovod distribution framework, which works in
concert with MPI. To train resnet-50 using 8 V100 GPUs, for example on DGX-1,
use the following command.

```
$ mpiexec --allow-run-as-root -np 8 python resnet.py --layers=50
                                                     --data_dir=/data/imagenet
                                                     --precision=fp16
                                                     --log_dir=/output/resnet50
```

Here we have assumed that imagenet is stored in tfrecord format in the directory
'/data/imagenet'. After training completes, evaluation is performed using the
validation dataset.

Some common training parameters can tweaked from the command line. Others must
be configured within the network scripts themselves.
