#!/bin/bash


# Usage: ethernet.sh server
# Usage: ethernet.sh client server_ip

# The iperf example:
#   https://www.veritas.com/support/en_US/article.000066021
# Determining PCIe numa node
#   https://superuser.com/questions/701090/determine-pcie-device-numa-node
#   Do something like that: 
#     cd /sys/class/net/eth0/device
#     where you will find numa_node, local_cpus, local_cpulist, the three 
#     files of interes to you. You can just cat them, and see the desired data.

. ./common.sh ../

interconnect=ethernet
os=host
server=
numa_node=0

port=45678
tm=30

image=
net=host
mb=100

help_message="\
Usage: $0 [options]\n\
\n\
Measures network bandwidth between two processes running on two remote hosts. Proceses can run on host node or inside Docker container. \
Ethernet and InfiniBand interconenct is supported. For Ethernet, iperf3 tool is used. For InfiniBand, ibv_rc_pingpong tool is used. \
This script runs remote process via ssh for current user. You need to be able to login on remote host with ssh without password. \
In Docker mode, script assumes that local and remote hosts provide specific Docker images. In particular, default image for Ethernet \
tests is 'hpe/benchmarks:ethernet' and for InfiniBand test it is 'hpe/benchmarks:infiniband'. This can be overwritten via command line \
arguments.\n\
All containers created by this script are autoremoved.\n\
  \n\
  options:\n\
    --interconnect infiniband | ethernet\n\
        Measure bandwidth of this interconnect. Default is 'ethernet'.\n\
    --os host | docker\n\
        Operating system to use. Default is 'host'.\n\
    --server [server]\n\
        Run server (the remote process) on this node. It can be either a node name or IP address.\n\
    --numa_node [numa_node]\n\
        Use this socket to bind server and client processes. The 'numactl' is used to bind. Default is 0.\n\
  \n\
  ethernet specific options:\n\
    --port [port]\n\
        Use this port for on server node. Default value is 45678\n\
    --tm [time]\n\
        Run experiment for this number of seconds. Default value is 30 seconds.\n\
  \n\
  docker specific options:\n\
    --image [image]\n\
       Use this image to run experiments. Default 'ethernet' image is 'hpe/benchmarks:ethernet'. Default 'infiniband' image is 'hpe/benchmarks:infiniband'.\n\
    --net bridge | host\n\
       Use this docker networking mechanism. If 'bridge' is selected, port mapping is used. Default is 'host'. See docker documentation on different options. This is only for Ethernet tests.\n\
    --mb [mb]\n\
       Ping pong this data volume (data size in megabytes). This is an InfiniBand specific option.\n\
  \n\
  \n\
  EXAMPLES:\n\
    $0 --interconnect ethernet --os host --server other_node_name\n\
       Run iperf3-based ethernet tests on host operating system. The server process will be run on 'other_node_name' node.\n\
    \n\
    $0 --interconnect ethernet --os docker --net host --server other_node_name --tm 60\n\
       Run iperf3-based ethernet tests inside docker container for 60 seconds. The server process will be run on 'other_node_name' node. The Docker networking is 'host'.\n\
    \n\
    $0 --interconnect infiniband --os host --server other_node_name --mb 10\n\
       Run ibv_rc_pingpong-based infiniband tests on host operating system. The server process will be run on 'other_node_name' node. Size of data being transfered is 10 mb.\n\
    \n\
    $0 --interconnect infiniband --os docker --server other_node_name --mb 20 --image hpe/infinband:latest --net bridge\n\
       Run ibv_rc_pingpong-based infiniband tests inside docker container. The server process will be run on 'other_node_name' node. Size of data being transfered is 20 mb. 'Bridge' network is used with port mapping. Custom Docker image (hpe/infinband:latest) is used to run tests.\n\
"

. $DLBS_ROOT/scripts/parse_options.sh

logvars "interconnect os server role port tm numa_node"

ssh_cmd="ssh -t -t $server"
numa_cmd="numactl --cpunodebind=$numa_node"
iperf_cmd=iperf3

# Figure out server and client command line arguments
if [ "$interconnect" = "ethernet"  ]; then
  server_cmd="$numa_cmd $iperf_cmd -s -p $port  -i 0 --one-off"
  client_cmd="$numa_cmd $iperf_cmd -c $server -t $tm -p $port -d -i 0"
  if [ "$os" = "host"  ]; then
    caption="Testing ethernet bandwidth on a host OS."
    server_exec="$ssh_cmd $server_cmd"
    client_exec="$client_cmd"
  else
    caption="Testing ethernet bandwidth inside docker container."
    image=${image:-hpe/benchmarks:ethernet}
    if [ "$net" = "host" ]; then
      docker_cmd="docker run --rm --net=$net -it $image"
    elif [ "$net" = "bridge" ]; then
      docker_cmd="docker run --rm --net=$net -p $port:$port -it $image"
    else
      logfatal "Unknown docker network ($net)"
    fi
    server_exec="$ssh_cmd $docker_cmd $server_cmd"
    client_exec="$docker_cmd $client_cmd"
  fi
elif [ "$interconnect" = "infiniband"  ]; then
  bytes=$(($mb*1024*1024))
  server_cmd="$numa_cmd ibv_rc_pingpong --size=$bytes"
  client_cmd="$numa_cmd ibv_rc_pingpong --size=$bytes $server"
  if [ "$os" = "host"  ]; then
    caption="Testing infiniband bandwidth on a host OS."
    server_exec="$ssh_cmd $server_cmd"
    client_exec="$client_cmd"
  else
    caption="Testing ethernet bandwidth inside docker container."
    image=${image:-hpe/benchmarks:infiniband}
    docker_cmd="docker run --net=host --privileged --rm -v /dev/infiniband:/dev/infiniband -ti $image"
    server_exec="$ssh_cmd $docker_cmd $server_cmd"
    client_exec="$docker_cmd $client_cmd"
  fi
else
  logfatal "Unknown network ($net)"
fi

# Run tests
loginfo $caption
logvars "server_exec client_exec"
# Run server on the remote machine
$server_exec > /dev/null &
server_pid=$!
sleep 2
# Run client on local machine
$client_exec
# Wait for server to finish.
# The 'iperf3' will automatically exit but not 'iperf'.
# The 'ibv_rc_pingpong' will exit automatically.
wait ${server_pid}
#kill -INT ${server_pid}
