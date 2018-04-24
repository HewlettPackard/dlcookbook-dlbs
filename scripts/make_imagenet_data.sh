#!/bin/bash

print_help() {
  format=$1
  human_readeable_format=$2
  frameworks=$3
  usage_example=$4

  echo "Usage: $0 $format OPTION..."
  echo "Generate dataset in $human_readeable_format format for $frameworks frameworks"
  echo ""
  echo "--input DIR                        Full path to a folder containing ImageNet dataset. The structure of the folder"
  echo "                                   should be standard. It should contain synsent directories. Those directories"
  echo "                                   should contain JPEG files. This folder should be writable."
  echo "--output DIR                       Output directory. This directory must not exist."
  echo "--docker_image IMAGE               A docker image to use. The script can only work in docker containers."
  echo "--docker (docker | nvidia-docker)  How to run docker containers - with 'docker' or 'nvidia-docker'. LMDB dataset"
  echo "                                   can be generated with both while others require nvidia-docker. [default: docker]"
  if [ ! "$format" == "tfrecord" ]; then
    echo "--img_size SIZE                    The size of images in dataset (SIZExSIZE). All images will be resized/cropped to"
    echo "                                   this size. [default: 300]"
  fi
  if [ "$format" == "lmdb" ]; then
    echo "--caffe_dir DIR                    A folder in docker container that contais Caffe's tool that covnerts ImageNet"
    echo "                                   (convert_imageset). It's usually \$CAFFE_ROOT/build/tools or \$CAFFE_INSTALL_PATH/bin."
    echo "                                   [default: /opt/caffe/bin]"
  fi
  if [ "$format" == "recordio" ]; then
    echo "--mxnet_dir DIR                    A folder in docker container that contais MXNET's tool that covnerts ImageNet"
    echo "                                   (im2rec). It's usually \$MXNET_ROOT/bin. [default: /opt/mxnet/bin]"
  fi
  if [[ "$format" =~ tfrecord ]]; then
    echo "--num_shards N                     Number of TFRecord files to create. Names of these files will be train-*****-of-*****."
    echo "                                   [default: 1]"
  fi
  if [ "$format" == "fast_tfrecord" ]; then
    echo "--num_workers K                    Number of workers (parallel jobs) to use to create dataset. Must be less than or equal"
    echo "                                   to number of GPUs. Also, num_shards % num_workers must be 0.  [default: 1]"
  fi
  if [ "$format" == "tfrecord" ]; then
    echo "--num_workers K                    Number of workers (parallel jobs) to use to create dataset. [default: 1]"
    echo "--bboxes_dir DIR                   A path to a folder with ImageNet original XML files containing bounding boxes."
  fi
  echo ""
  echo "Usage:"
  echo "  $usage_example"
}

if [[ "$1" =~ ^(help|-h)$ ]]; then
  if [ "$2" == "lmdb" ]; then
    print_help "lmdb" "LMDB" "Caffe, Caffe2 and PyTorch" \
               "$0 --dataset lmdb --input /storage/imagenet --output /storage/lmdb --docker_image hpe/bvlc_caffe:cuda9-cudnn7 --img_size 300 --docker docker --caffe_dir /opt/caffe/bin"
  elif [ "$2" == "recordio" ]; then
    print_help "recordio" "RecordIO" "MXNET" \
               "$0 --dataset recordio --input /storage/imagenet --output /storage/recordio --docker_image hpe/mxnet:cuda9-cudnn7 --img_size 300 --docker nvidia-docker --mxnet_dir /opt/mxnet/bin"
  elif [ "$2" == "tfrecord" ]; then
    print_help "tfrecord" "TFRecord" "TensorFlow" \
               "$0 --dataset tfrecord --input /storage/imagenet --output /storage/tfrecord --docker_image --docker_image hpe/tensorflow:cuda9-cudnn7 --docker nvidia-docker --bboxes_dir /storage/bboxes --num_shards 10 --num_workers 10"
  elif [ "$2" == "fast_tfrecord" ]; then
    print_help "fast_tfrecord" "Fast TFRecord" "TensorFlow (tf_cnn_benchmarks version from DLBS)" \
               "$0 --dataset fast_tfrecord --input /storage/imagenet --output /storage/fast_tfrecord --docker_image hpe/tensorflow:cuda9-cudnn7 --img_size 300 --docker nvidia-docker --num_shards 24 --num_workers 8"
  else
    echo "Generate benchmark dataset based on ImageNet data. The tool can generate"
    echo "datasets in the following formats:"
    echo "--------------------------------------------------"
    echo "| Format        | Frameworks                     |"
    echo "|---------------|--------------------------------|"
    echo "| lmdb          | Caffe, Caffe2 , PyTorch        |"
    echo "| recordio      | MXNET                          |"
    echo "| tfrecord      | TensorFlow                     |"
    echo "| fast_tfrecord | TensorFlow (tf_cnn_benchmarks) |"
    echo "--------------------------------------------------"
    echo "Get more help: "
    echo "   $0 help FORMAT"
    echo "where FORMAT is one of [lmdb, recordio, tfrecord, fast_tfrecord]"
  fi
  exit 0
fi

# Common parameters
dataset=          # Type of dataset (tfrecord, fast_tfrecord, recordio, lmdb)
input=            # Folder with input ImageNet JPEGs
output=           # Output folder
docker_image=     # Docker image to use, dataset specific
img_size=300      # Resize images to this size if applicable
docker=docker     # How to run docker: docker or nvidia-docker

# TensorFlow(TF_CNN_BENCHMARKS): tfrecord parameters
bboxes_dir=
# TensorFlow(TF_CNN_BENCHMARKS): fast_tfrecord parameters
num_shards=1
num_workers=1
# MXNET: recordio parameters
mxnet_dir=/opt/mxnet/bin
# Caffe, Caffe2, PyTorch: LMDB parameters
caffe_dir=/opt/caffe/bin         # Two options in general
                                 #    /opt/caffe/bin
                                 #    /opt/caffe/build/tools
# ./scripts/make_imagenet_data.sh --docker_image hpe/bvlc_caffe:cuda8-cudnn6 --input /fdata/serebrya/imagenet/data/train --output /fdata/serebrya/imagenet/data/lmdb  --dataset lmdb --img_size 300

unknown_params_action=set
. ./scripts/environment.sh
. $DLBS_ROOT/scripts/parse_options.sh || exit 1;

assert_dirs_exist $input
assert_docker_img_exists $docker_image
[ -d "$output" ] && logfatal "Output directory ($output) must not exist."

imagenet_tools=$DLBS_ROOT/python/dlbs/data/imagenet/imagenet_tools.py
if [ "$dataset" == "lmdb" ]; then
  # Check that database directory does not exist but it's parent does exist
  db_dir=$(dirname "$output")
  db_name=$(basename "$output")
  assert_dirs_exist $db_dir
  # Generate if not present a textual file that maps a relative image file name to its label
  docker_args="--rm -ti --volume=${input}:/imagenet/input --volume=${db_dir}:/imagenet/output ${docker_image}"
  [ ! -f "${input}/caffe_labels.txt" ] && python $imagenet_tools "build_caffe_labels" "$input" "${input}/caffe_labels.txt"
  assert_files_exist "${input}/caffe_labels.txt"
  # Build docker and converter arguments
  converter="${caffe_dir}/convert_imageset"
  script="GLOG_logtostderr=1 ${converter} --resize_height=${img_size} --resize_width=${img_size} --shuffle /imagenet/input/ /imagenet/input/caffe_labels.txt /imagenet/output/${db_name}"
elif [ "$dataset" == "recordio" ]; then
  mkdir -p $output
  # Generate if not present a textual file that maps a relative image file name to its label
  [ ! -f "${input}/mxnet_labels.txt" ] && python $imagenet_tools "build_mxnet_labels" "$input" "${input}/mxnet_labels.txt"
  assert_files_exist "${input}/mxnet_labels.txt"
  # Build docker and converter arguments
  docker_args="--rm -ti --volume=${input}:/imagenet/input --volume=${output}:/imagenet/output ${docker_image}"
  converter="${mxnet_dir}/im2rec"
  script="$converter /imagenet/input/mxnet_labels.txt /imagenet/input/ /imagenet/output/recordio.bin resize=${img_size}"
elif [ "$dataset" == "tfrecord" ]; then
  mkdir -p $output
  # Build the file with bounding boxes. You need to have original ImageNet, or just
  # folder with XML files.
  if [ ! -f "${input}/tensorflow_bboxes.csv" ]; then
    [ ! -f "${bboxes_dir}" ] && logfatal "Please, provide path to directory with bounding boxes in XML format with --bboxes_dir parameter."
    python $DLBS_ROOT/python/dlbs/data/imagenet/tensorflow_process_bboxes.py ${bboxes_dir} > ${input}/tensorflow_bboxes.csv
  fi
  # Build files with sybsets and human readeable labels
  for fid in "tensorflow_synsets" "tensorflow_human_labels"
  do
    [ ! -f "${input}/$fid.txt" ] && python $imagenet_tools "build_$fid" "$input" "${input}/$fid.txt"
    assert_files_exist "${input}/$fid.txt"
  done
  # Build docker and converter arguments
  docker_args="--rm -ti --volume=${DLBS_ROOT}:/workspace --volume=${input}:/imagenet/input --volume=${output}:/imagenet/output ${docker_image}"
  converter="/workspace/python/dlbs/data/imagenet/tensorflow_build_imagenet_data.py"
  script="PYTHONPATH=/workspace/python python \
          $converter --train_directory /imagenet/input --train_shards $num_shards --output_directory /imagenet/output \
                     --validation_shards 0 --num_threads $num_workers \
                     --imagenet_metadata_file /imagenet/input/tensorflow_human_labels.txt \
                     --bounding_box_file /imagenet/input/tensorflow_bboxes.csv \
                     --labels_file /imagenet/input/tensorflow_synsets.txt"
elif [ "$dataset" == "fast_tfrecord" ]; then
  mkdir -p $output
  # Build docker and converter arguments
  docker_args="--rm -ti --volume=${DLBS_ROOT}:/workspace --volume=${input}:/imagenet/input --volume=${output}:/imagenet/output ${docker_image}"
  converter="/workspace/python/dlbs/data/imagenet/tensorflow_data.py"
  script="PYTHONPATH=/workspace/python python $converter --input_dir /imagenet/input --num_shards $num_shards --output_dir /imagenet/output --num_workers $num_workers --img_size $img_size"
else
  logfatal "Invalid dataset: ${dataset}"
fi

# Convert
loginfo "Docker command: $docker run ${docker_args}"
loginfo "Converter command: $script"
$docker run ${docker_args} /bin/bash -c "$script"
