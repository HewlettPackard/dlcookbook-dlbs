Bootstrap: localimage
FROM: /var/lib/SingularityImages/tensorflow-!{TF_VERSION}-!{CUDA_VERSION}-!{CUDNN_VERSION}.img

%labels
	Maintainer Stephen Fleischman
    Framework TensorFlow
    Version  !{TF_VERSION}
    Build  CUDA !{CUDA_VERSION} cuDNN ${CUDNN_VERSION} x86_64 AVX2 (Broadwell), OFED IB.
    Installed Horovod, OpenNMT and Tensor2Tensor

%help
    TensorFlow !{TF_VERSION} GPU Singularity Container
    Maintainer: Stephen Fleischman

%post
    export PATH=/opt/anaconda3/bin:/usr/local/cuda-!{CUDA_VERSION}/bin:$PATH
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

    add-apt-repository -y ppa:ubuntugis/ppa && apt-get update
    apt-get install -y libtinfo-dev libncurses-dev
    apt-get install -y gdal-bin


    python3 -m pip install rasterio geopandas folium cartopy rio-toa rio-l8qa
    conda install -y -c conda-forge arcsi python-fmask tuiview ncurses=6 gdal

    ldconfig -v

%runscript
    echo "Singularity Container: TensorFlow !{TF_VERSION}, Ubuntu !{UBUNTU_VERSION}, CUDA !{CUDA_VERSION}, cuDNN ${CUDNN_VERSION}, Anaconda Python !{PYTHON_VERSION}, AVX2 instructions."
    echo "The image contains: Jupyter, Horovod, NVidia examples, OpenNMT for TensorFlow, Tensor2Tensor, GeoSpatial Python packages."
