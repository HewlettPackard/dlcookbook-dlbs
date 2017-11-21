# __Docker__

We have chosen Docker containers as a primary mechanism to run benchmarks. There are several reasons for that:

1. Minimal intervention and dependency requirements to a machine where benchmarks run. There are many different hardware most of which we do not control, have regular access etc. Those machines are managed by other teams and we do not want to install/reinstall packages that are only required for benchmarks/experiments.
2. Benchmarks are reproducible. Setup environment once, run same code in same environment everywhere on variety of hardware.
3. Same performance as benchmarking bare metal frameworks.
4. Ease of use. Clone repository, build/pull images, run script. Get results, upload them to github. Share, discuss, analyze. The setup stage is minimized and focus shifts to results analysis.
5. Use various instances of the same framework (version/compilation flags ...) transparently. They all differ by a container identifier. Everything else remains the same.  
6. Throw away SW installation routines that may take days. Git rid of dependency problems. Applications run in sandbox environments - they will never pick the wrong library at runtime - save hours of debugging for some other useful activities.
7. Push new version of your container to your registry. It then immediately becomes available to all hosts in a cluster.

We have gathered several useful URLs/guides on how to isntall/troublechoot docker installation. We also describe how to build/pull docker images.
* [Read](/docker/install_docker.md?id=installing-docker) about installing/troubleshooting docker-related software.
* [Read](/docker/docker_network.md?id=docker-networking) about using network (Ethernet/InfiniBand) inside docker containers.
* [Read](/docker/pull_build_images.md?id=buildpull-docker-images) about pulling/building framework images.
