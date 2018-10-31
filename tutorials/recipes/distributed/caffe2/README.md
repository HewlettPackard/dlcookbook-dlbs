# Distributed Caffe2 benchmarks
Before running examples below, make sure:

1. You have docker/nvidia-docker installed on each node.
2. Clone DLBS.
3. Pull Caffe2 docker image from NGC on each node:
   ```bash
   docker pull nvcr.io/nvidia/caffe2:18.05-py2
   ```

### Running distributed benchmarks on multiple nodes
Example scripts demonstrate how to run distributed Cafe2 benchmarks
with DLBS.

1. Update `config.json` file, in particular:
   ```bash
   # Change number of nodes you plan to use.
   "exp.num_nodes": 2
   # Change rendezvous location. Two methods are supported. The first one is a
   # file system based rendezvous. It is specified with 'file://' schema. The
   # second one is a redis database based rendezvous. The format is redis://HOST:PORT.
   "caffe2.rendezvous": "file:///some/shared/path"
   # Change network interface to use for communications (run ifconfig)
   "caffe2.interface": "ib0"
   # Change GPUs to use on each node.
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
   Read comments in `run` script to learn more details. Node with rank 0 must be
   started first.
