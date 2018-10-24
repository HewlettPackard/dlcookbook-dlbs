# Distributed PyTorch benchmarks
Before running examples below, make sure:

1. You have docker/nvidia-docker installed on each node.
2. Clone DLBS.
3. Pull PyTorch docker image from NGC on each node:
   ```bash
   docker pull nvcr.io/nvidia/pytorch:18.06-py3
   ```

### Running distributed benchmarks on multiple nodes
Example scripts demonstrate how to run distributed PyTorch benchmarks
with DLBS.

1. Update `config.json` file, in particular:
   ```bash
   # Change number of nodes you plan to use.
   "exp.num_nodes": 1
   # Change rendezvous location. This should be one of the nodes.
   "pytorch.distributed_rendezvous": "127.0.0.1:29500"
   # Change number of GPUs to use on each node. This config works for
   # an 8-GPU server.
   "exp.gpus": "0,1,2,3,4,5,6,7"
   ```
2. On each node, run the `run` script providing node rank (0-based). For a two node
   benchmark, run the following:
   ```bash
   # Node A
   ./run 0
   # Node B
   ./run 1
   ```
Make sure that you have DLBS on each node installed, check docker image that must
present on each node.


### Running distributed benchmarks on a single nodes
You can try to run distributed PyTorch on a single node. The following needs to
be done (this is the example for a 8-GPU node):

1. Update `config.json` file, in particular:
   ```bash
   # Change number of nodes you plan to use. We'll use virtual nodes.
   "exp.num_nodes": 2
   # Change rendezvous location. Localhost will work.
   "pytorch.distributed_rendezvous": "127.0.0.1:29500"
   # We need to provide unique GPUs for each virtual node. We'll do that in run
   # script, so, here we need to either remove `exp.gpus` parameter or make it
   # empty.
   "exp.gpus": ""
   ```

2. Create to copies of the `run` script, let's say `node01` and `node02`. Update
   them in the following way:
   ```bash
   # File node01
   python $experimenter run --config=./config.json -Ppytorch.distributed_rank=0 -Pexp.gpus='"0,1,2,3"'
   # File node02
   python $experimenter run --config=./config.json -Ppytorch.distributed_rank=1 -Pexp.gpus='"4,5,6,7"'
   ```

### Running distributed benchmarks with real data
There is no special support for real data now for distributed training. The
parameter `exp.data_dir` that points to a dataset (LMDB) can be used. All
processes will read the very same data. That's probably OK for performance
benchmarks. I'll update the code in future version.
