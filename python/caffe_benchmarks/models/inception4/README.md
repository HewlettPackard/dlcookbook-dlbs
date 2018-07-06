# __InceptionV4__

Is based on this [descriptor](https://github.com/soeaver/caffe-model/blob/master/cls/inception/deploy_inception-v4.prototxt).
Changes:

1. A name section.
2. Parameters of pooling layer `pool_8x8_s1` that implements global pooling changed
   from `global_pooling: true` to specifying kernel and stride:
```
pooling_param {
    pool: AVE
    kernel_size: 8
    stride: 1
}
```
to make it work with TensorRT. See this [thread](https://devtalk.nvidia.com/default/topic/985607/tensorrt-1-0-fails-on-squeezenet/)
for more details.

## Visualization:

1. http://ethereon.github.io/netscope/#/gist/8fdab7a3ea5bceb9169832dfd73b5e31
