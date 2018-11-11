model: densenet169
model_title: densenet169
model_url: https://raw.githubusercontent.com/shicai/DenseNet-Caffe/a68651c0b91d8dcb7c0ecd39d1fc76da523baf8a/DenseNet_169.prototxt
updates: softmax layer
layer {
  name: "prob"
  type: "Softmax"
  bottom: "fc6"
  top: "prob"
}
