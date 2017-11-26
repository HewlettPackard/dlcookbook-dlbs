#!/bin/bash

# This script generates residual units for ResNet200 ans ResNet269 models.
# The base model is ResNet152.

# Residual Unit for ResNets with number of layers >= 50
# 4 stages in every network. Every stage has specific number of residual blocks
# all having same output number of filters (channels):
#  stage 1: 256
#  stage 2: 512
#  stage 3: 1024
#  stage 4: 2048
# Internal number of filters is the above numbers / 4.

# Template for residual unit. These needs to be replaced:
#   __BOTTOM_BLOB__             Identifier of a bottom blob i.e. res3b7.
#   __NUM_OUTPUT_CHANNELS__     Number of output channels (see above).
#   __NUM_INTERNAL_CHANNELS__   Number of internal channels (__NUM_OUTPUT_CHANNELS__ / 4).

#  __UNIT_ID__                  Identifier of this unit i.e. 3b8
unit_template=$(cat <<-END
layer {
  name: "res__UNIT_ID___branch2a"
  type: "Convolution"
  bottom: "__BOTTOM_BLOB__"
  top: "res__UNIT_ID___branch2a"
  convolution_param {
    num_output: __NUM_INTERNAL_CHANNELS__
    bias_term: false
    pad: 0
    kernel_size: 1
    stride: 1
  }
}
layer {
  name: "bn__UNIT_ID___branch2a"
  type: "BatchNorm"
  bottom: "res__UNIT_ID___branch2a"
  top: "res__UNIT_ID___branch2a"
  batch_norm_param {
    use_global_stats: true
  }
}
layer {
  name: "scale__UNIT_ID___branch2a"
  type: "Scale"
  bottom: "res__UNIT_ID___branch2a"
  top: "res__UNIT_ID___branch2a"
  scale_param {
    bias_term: true
  }
}
layer {
  name: "res__UNIT_ID___branch2a_relu"
  type: "ReLU"
  bottom: "res__UNIT_ID___branch2a"
  top: "res__UNIT_ID___branch2a"
}
layer {
  name: "res__UNIT_ID___branch2b"
  type: "Convolution"
  bottom: "res__UNIT_ID___branch2a"
  top: "res__UNIT_ID___branch2b"
  convolution_param {
    num_output: __NUM_INTERNAL_CHANNELS__
    bias_term: false
    pad: 1
    kernel_size: 3
    stride: 1
  }
}
layer {
  name: "bn__UNIT_ID___branch2b"
  type: "BatchNorm"
  bottom: "res__UNIT_ID___branch2b"
  top: "res__UNIT_ID___branch2b"
  batch_norm_param {
    use_global_stats: true
  }
}
layer {
  name: "scale__UNIT_ID___branch2b"
  type: "Scale"
  bottom: "res__UNIT_ID___branch2b"
  top: "res__UNIT_ID___branch2b"
  scale_param {
    bias_term: true
  }
}
layer {
  name: "res__UNIT_ID___branch2b_relu"
  type: "ReLU"
  bottom: "res__UNIT_ID___branch2b"
  top: "res__UNIT_ID___branch2b"
}
layer {
  name: "res__UNIT_ID___branch2c"
  type: "Convolution"
  bottom: "res__UNIT_ID___branch2b"
  top: "res__UNIT_ID___branch2c"
  convolution_param {
    num_output: __NUM_OUTPUT_CHANNELS__
    bias_term: false
    pad: 0
    kernel_size: 1
    stride: 1
  }
}
layer {
  name: "bn__UNIT_ID___branch2c"
  type: "BatchNorm"
  bottom: "res__UNIT_ID___branch2c"
  top: "res__UNIT_ID___branch2c"
  batch_norm_param {
    use_global_stats: true
  }
}
layer {
  name: "scale__UNIT_ID___branch2c"
  type: "Scale"
  bottom: "res__UNIT_ID___branch2c"
  top: "res__UNIT_ID___branch2c"
  scale_param {
    bias_term: true
  }
}
layer {
  name: "res__UNIT_ID__"
  type: "Eltwise"
  bottom: "__BOTTOM_BLOB__"
  bottom: "res__UNIT_ID___branch2c"
  top: "res__UNIT_ID__"
}
layer {
  name: "res__UNIT_ID___relu"
  type: "ReLU"
  bottom: "res__UNIT_ID__"
  top: "res__UNIT_ID__"
}
END
)

# get_unit bottom_blob unit_id num_internal_channels num_output_channels
get_unit() {
  [ "$#" -ne 4 ] && logfatal "get_unit: 4 arguments expected (bottom_blob unit_id num_internal_channels num_output_channels)";

  local unit=${unit_template//__BOTTOM_BLOB__/$1}
  unit=${unit//__UNIT_ID__/$2}
  unit=${unit//__NUM_INTERNAL_CHANNELS__/$3}
  unit=${unit//__NUM_OUTPUT_CHANNELS__/$4}

  echo "${unit}"
}

# This generates missing residual units for ResNet200 for 2nd stage
# assuming base config is ResNet152. One file will be created - resnet200_2.txt
resnet_200() {
  local out_file="./resnet200_2.txt"
  [ -f "${out_file}" ] && rm -f ${out_file}
  for i in {8..23}; do
    local unit=$(get_unit 3b$(($i-1)) 3b${i} 128 512)
    echo "$unit" >> ${out_file}
  done
}

# This generates missing residual units for ResNet269 for 2nd, 3rd and 4th stages
# assuming base config is ResNet152. Three files will be created - resnet269_2.txt,
# resnet269_3.txt and resnet269_4.txt.
resnet_269() {
  # 2nd stage
  local out_file="./resnet269_2.txt"
  [ -f "${out_file}" ] && rm -f ${out_file}
  for i in {8..29}; do
    local unit=$(get_unit res3b$(($i-1)) 3b${i} 128 512)
    echo "$unit" >> ${out_file}
  done
  # 3rd stage
  local out_file="./resnet269_3.txt"
  [ -f "${out_file}" ] && rm -f ${out_file}
  for i in {36..47}; do
    local unit=$(get_unit res4b$(($i-1)) 4b${i} 256 1024)
    echo "$unit" >> ${out_file}
  done
  # 4th stage. A bit different due to different naming notation.
  # Currently  - 'res5b' and 'res5c'. We need to add 'd', 'e', 'f', 'g' and 'h'
  local out_file="./resnet269_4.txt"
  [ -f "${out_file}" ] && rm -f ${out_file}
  echo "$(get_unit res5c 5d 512 2048)" >> ${out_file}
  echo "$(get_unit res5d 5e 512 2048)" >> ${out_file}
  echo "$(get_unit res5e 5f 512 2048)" >> ${out_file}
  echo "$(get_unit res5f 5g 512 2048)" >> ${out_file}
  echo "$(get_unit res5g 5h 512 2048)" >> ${out_file}
}

#resnet_200
resnet_269
