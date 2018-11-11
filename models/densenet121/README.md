model: densenet121
model_title: DenseNet121
model_url: https://raw.githubusercontent.com/shicai/DenseNet-Caffe/a68651c0b91d8dcb7c0ecd39d1fc76da523baf8a/DenseNet_121.prototxt
updates: softmax layer:
layer {
  name: "prob"
  type: "Softmax"
  bottom: "fc6"
  top: "prob"
}
