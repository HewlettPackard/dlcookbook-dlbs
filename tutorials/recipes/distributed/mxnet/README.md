# Distributed MXNET benchmarks
Before running examples below, make sure:

1. You have docker/nvidia-docker installed on each node.
2. Clone DLBS.
3. Build MXNET docker image:
   ```bash
   cd dlbs/docker
   ./build.sh mxnet/18.11
   ```
   Normally, we recommend using containers from NGC. However, NGC mxnet containers
   are built without distributed KVSTORE (parameter server) support.
   You do not need to build this docker images on every node. Build it on
   one node, then save it to disk and load on other nodes:
   ```bash
   # node A
   docker save dlbs/mxnet:18.11 > dlbs_mxnet:18.11
   # node B
   docker load --input dlbs_mxnet:18.11
   ```

### Running distributed benchmarks on multiple nodes
Example scripts demonstrate how to run distributed MXNET benchmarks
with DLBS.

1. Update `config.json` file, in particular:
   ```bash
   # Change number of nodes you plan to use.
   "exp.num_nodes": 2
   # Change rendezvous location. This should be one of the nodes (rank 0).
   # The INTERFACE is optional here (read about DMLC_INTERFACE), so it can be
   # just IP_ADDRESS:29500. (InfiniBand (ib0) will significantly accelerate
   # training).
   "mxnet.rendezvous": "INTERFACE:IP_ADDRESS:29500"
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
   Pay attention to the following. The script will run scheduler on node with rank 0.
   So, make sure that you run rank 0 on node that is specified as rendezvous location.

### How does DLBS run distributed MXNET?
DLBS will run on each node one worker and one server processes. On node with rank 0 DLBS will additionally run scheduler process. So:
  1. There's always one scheduler running on node with rank 0.
  2. On each node there's one worker and one server. One worker can use as many GPU as it needs.

DLBS will define the following environmental variables:   `DMLC_ROLE`, `DMLC_PS_ROOT_URI`, `DMLC_PS_ROOT_PORT`, `DMLC_NUM_SERVER`, `DMLC_NUM_WORKER`, `PS_VERBOSE`. If user has provided a network interface, additionally `DMLC_INTERFACE` will be defined.
