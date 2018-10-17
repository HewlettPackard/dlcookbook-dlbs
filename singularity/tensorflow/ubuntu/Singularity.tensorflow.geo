Bootstrap: localimage
FROM: /var/lib/SingularityImages/tensorflow-!{TF_VERSION}.img

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

    # Geo stuff
    mkdir -p /tmp/install
    cd /tmp/install && curl -sL http://download.osgeo.org/geos/geos-3.7.0.tar.bz2|tar xvj 
    cd /tmp/install/geos-3.7.0 && ./configure && make && make install
    cd /tmp/install && curl -sL http://download.osgeo.org/proj/proj-4.9.1.tar.gz|tar xvz
    cd /tmp/install/proj-4.9.1 && ./configure && make && make install
    cd /tmp/install && curl -sL http://download.osgeo.org/libtiff/tiff-4.0.9.tar.gz|tar xvz
    cd /tmp/install/tiff-4.0.9 && ./configure && make && make install
    cd /tmp/install && curl -sL http://download.osgeo.org/geotiff/libgeotiff/libgeotiff-1.4.2.tar.gz|tar xvz
    cd /tmp/install/libgeotiff-1.4.2 && cmake ./CMakeLists.txt && make install 
    cd /tmp/install && rm -rf pycpt && git clone https://github.com/j08lue/pycpt.git
    cd /tmp/install/pycpt &&  python ./setup.py install
    pip install --no-binary shapely shapely
    python3 -m pip install rasterio geopandas folium cartopy rio-toa rio-l8qa

    conda install -c conda-forge arcsi
    conda install -c conda-forge python-fmask
    conda install -c conda-forge tuiview
    conda install -c conda-forge scikit-learn matplotlib h5py
    conda update -c conda-forge --all

    ldconfig -v
    cd /tmp && rm -rf *

%runscript
    echo "Singularity Container: TensorFlow !{TF_VERSION}, Ubuntu !{UBUNTU_VERSION}, CUDA !{CUDA_VERSION}, cuDNN ${CUDNN_VERSION}, Anaconda Python !{PYTHON_VERSION}, AVX2 instructions."
    echo "The image contains: Jupyter, Horovod, NVidia examples, OpenNMT for TensorFlow, Tensor2Tensor, GeoSpatial Python packages."
