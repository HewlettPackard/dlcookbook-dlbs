# __InceptionV3__

Is based on this [descriptor](https://github.com/soeaver/caffe-model/blob/master/cls/inception/deploy_inception-v3.prototxt).
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

1. http://ethereon.github.io/netscope/#/gist/04a797f778a7d513a9b52af4c1dbee4e
2. https://4.bp.blogspot.com/-TMOLlkJBxms/Vt3HQXpE2cI/AAAAAAAAA8E/7X7XRFOY6Xo/s1600/image03.png
