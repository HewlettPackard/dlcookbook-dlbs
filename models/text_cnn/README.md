## Text CNN

Text classification with simple convolution neural networks.

#### Caffe2 implementation notes

That's not a problem to implement this network in Caffe. One thing to keep in mind implementing this topology in Caffe2 is that convolution and pooling operations have non square kernels. In particular, kernels of convolution operations have shape `[N, EmbeddingSize]` where `N` is the size of word window (for instance 1,3,5) and `EmbeddingSize` is the dimension of word embedding space. Kernel for pooling operation has shape `[M, 1]` where `M` is the maximal sentence length i.e. we do max pooling in time dimension.

In Caffe2 CNN Model wrapper does not allow to specify non square kernels' shapes. I have figured out how to rewrite `Conv` method to support this. Still need to figure out how to do the same for pooling operation.

Now, this network works in Caffe2 only when loading from file. Code implementation is incomplete and must not be used.

#### References

[1] https://arxiv.org/pdf/1408.5882.pdf

[2] http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/

[3] https://github.com/dennybritz/cnn-text-classification-tf/blob/master/train.py
