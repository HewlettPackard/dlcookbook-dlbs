# __Docker networking__

## Introduction
A good introduction to docker networking is this [slide deck](https://www.slideshare.net/lorispack/docker-networking-101):

1. `Bridge network`. Slides 10-11. This is a default option. Can be used for containers hosted within same node.
2. `Port mapping with bridge network`. Slides 12-13. Can be used for containers hosted on multiple nodes.
3. `Host network`. Slide 14. Can be used for multi-host networking. Gives full access of the host network to the container using `--net=host` option.

We provide a script in the [scripts](https://github.com/HewlettPackard/dlcookbook-dlbs/tree/master/scripts) folder named [test_bandwidth.sh](https://github.com/HewlettPackard/dlcookbook-dlbs/blob/master/scripts/test_bandwidth.sh) that can be used to measure ethernet and infiniband bandwidth between two nodes for host and docker workloads. This is the `--help` output of this script:

```bash
Usage: ./test_bandwidth.sh [options]

Measures network bandwidth between two processes running on two remote hosts. Proceses can run on host node or inside Docker container. Ethernet and InfiniBand interconenct is supported. For Ethernet, iperf3 tool is used. For InfiniBand, ibv_rc_pingpong tool is used. This script runs remote process via ssh for current user. You need to be able to login on remote host with ssh without password. In Docker mode, script assumes that local and remote hosts provide specific Docker images. In particular, default image for Ethernet tests is 'hpe/benchmarks:ethernet' and for InfiniBand test it is 'hpe/benchmarks:infiniband'. This can be overwritten via command line arguments.
All containers created by this script are autoremoved.

  options:
    --interconnect infiniband | ethernet
        Measure bandwidth of this interconnect. Default is 'ethernet'.
    --os host | docker
        Operating system to use. Default is 'host'.
    --server [server]
        Run server (the remote process) on this node. It can be either a node name or IP address.
    --numa_node [numa_node]
        Use this socket to bind server and client processes. The 'numactl' is used to bind. Default is 0.

  ethernet specific options:
    --port [port]
        Use this port for on server node. Default value is 45678
    --tm [time]
        Run experiment for this number of seconds. Default value is 30 seconds.

  docker specific options:
    --image [image]
       Use this image to run experiments. Default 'ethernet' image is 'hpe/benchmarks:ethernet'. Default 'infiniband' image is 'hpe/benchmarks:infiniband'.
    --net bridge | host
       Use this docker networking mechanism. If 'bridge' is selected, port mapping is used. Default is 'host'. See docker documentation on different options. This is only for Ethernet tests.
    --mb [mb]
       Ping pong this data volume (data size in megabytes). This is an InfiniBand specific option.


  EXAMPLES:
    ./test_bandwidth.sh --interconnect ethernet --os host --server other_node_name
       Run iperf3-based ethernet tests on host operating system. The server process will be run on 'other_node_name' node.

    ./test_bandwidth.sh --interconnect ethernet --os docker --net host --server other_node_name --tm 60
       Run iperf3-based ethernet tests inside docker container for 60 seconds. The server process will be run on 'other_node_name' node. The Docker networking is 'host'.

    ./test_bandwidth.sh --interconnect infiniband --os host --server other_node_name --mb 10
       Run ibv_rc_pingpong-based infiniband tests on host operating system. The server process will be run on 'other_node_name' node. Size of data being transfered is 10 mb.

    ./test_bandwidth.sh --interconnect infiniband --os docker --server other_node_name --mb 20 --image hpe/infinband:latest --net bridge
       Run ibv_rc_pingpong-based infiniband tests inside docker container. The server process will be run on 'other_node_name' node. Size of data being transfered is 20 mb. 'Bridge' network is used with port mapping. Custom Docker image (hpe/infinband:latest) is used to run tests.
```

This script provides this basic functionality:

1. Demonstrates how to use `port mapping` and `host` networking for multi-host docker applications. This does not hurt networking performance.
2. Demonstrates how to use InfiniBand inside Docker containers. This does not hurt networking performance.
3. We use `ssh` to login to remote hosts and run processes there. This approach is now used to run distributed Docker-based workloads.

## Performance
We provide example performance for two machines (Apollo6000) interconnected with Ethernet and InfiniBand (unidirectional bandwidth):

1. `Ethernet`
   1. `Host-Host`: 942 Mbits/sec
   2. `Container-Container with port mapping`: 942 Mbits/sec
   3. `Container-Container with host networking`: 942 Mbits/sec
2. `Infiniband`
   1. `Host-Host`: 50517.71 Mbit/sec
   2. `Container-Container`: 50519.24 Mbit/sec

## References

[1] [Docker networking concepts](https://github.com/docker/labs/tree/master/networking/concepts)

[2] [Deep dive into Docker 1.12 Networking](https://www.youtube.com/watch?v=nXaF2d97JnE)

[3] [Docker container networking user guide](https://docs.docker.com/engine/userguide/networking/)
