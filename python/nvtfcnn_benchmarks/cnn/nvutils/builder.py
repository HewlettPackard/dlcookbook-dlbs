import tensorflow as tf

class LayerBuilder(object):
    def __init__(self, activation=None, data_format='channels_last',
                 training=False, use_batch_norm=False, batch_norm_config=None):
        self.activation        = activation
        self.data_format       = data_format
        self.training          = training
        self.use_batch_norm    = use_batch_norm
        self.batch_norm_config = batch_norm_config
        if self.batch_norm_config is None:
            self.batch_norm_config = {
                'decay':   0.9,
                'epsilon': 1e-4,
                'scale':   True,
                'zero_debias_moving_mean': False,
            }

    def _conv2d(self, inputs, activation, separable, *args, **kwargs):
        conv_func = tf.layers.separable_conv2d if separable else tf.layers.conv2d
        x = conv_func(
                inputs, data_format=self.data_format,
                use_bias=not self.use_batch_norm,
                activation=None if self.use_batch_norm else activation,
                *args, **kwargs)
        if self.use_batch_norm:
            x = self.batch_norm(x)
            x = activation(x) if activation is not None else x
        return x

    def conv2d_linear(self, inputs, *args, **kwargs):
        return self._conv2d(inputs, None, False, *args, **kwargs)

    def conv2d(self, inputs, *args, **kwargs):
        return self._conv2d(inputs, self.activation, False, *args, **kwargs)

    def separable_conv2d_linear(self, inputs, *args, **kwargs):
        return self._conv2d(inputs, None, True, *args, **kwargs)

    def separable_conv2d(self, inputs, *args, **kwargs):
        return self._conv2d(inputs, self.activation, True, *args, **kwargs)

    def pad2d(self, inputs, begin, end=None):
        if end is None:
            end = begin
        try: _ = begin[1]
        except TypeError: begin = [begin, begin]
        try: _ = end[1]
        except TypeError: end = [end, end]
        if self.data_format == 'channels_last':
            padding = [[0, 0], [begin[0], end[0]], [begin[1], end[1]], [0, 0]]
        else:
            padding = [[0, 0], [0, 0], [begin[0], end[0]], [begin[1], end[1]]]
        return tf.pad(inputs, padding)

    def max_pooling2d(self, inputs, *args, **kwargs):
        return tf.layers.max_pooling2d(
            inputs, data_format=self.data_format, *args, **kwargs)

    def average_pooling2d(self, inputs, *args, **kwargs):
        return tf.layers.average_pooling2d(
            inputs, data_format=self.data_format, *args, **kwargs)

    def dense_linear(self, inputs, units, **kwargs):
        return tf.layers.dense(inputs, units, activation=None)

    def dense(self, inputs, units, **kwargs):
        return tf.layers.dense(inputs, units, activation=self.activation)

    def activate(self, inputs, activation=None):
        activation = activation or self.activation
        return activation(inputs) if activation is not None else inputs

    def batch_norm(self, inputs, **kwargs):
        all_kwargs = dict(self.batch_norm_config)
        all_kwargs.update(kwargs)
        data_format = 'NHWC' if self.data_format == 'channels_last' else 'NCHW'
        return tf.contrib.layers.batch_norm(
            inputs, is_training=self.training, data_format=data_format,
            fused=True, **all_kwargs)

    def spatial_average2d(self, inputs):
        shape = inputs.get_shape().as_list()
        if self.data_format == 'channels_last':
            n, h, w, c = shape
        else:
            n, c, h, w = shape
        n = -1 if n is None else n
        x = tf.layers.average_pooling2d(inputs, (h, w), (1, 1),
                                        data_format=self.data_format)
        return tf.reshape(x, [n, c])

    def flatten2d(self, inputs):
        x = inputs
        if self.data_format != 'channel_last':
            # Note: This ensures the output order matches that of NHWC networks
            x = tf.transpose(x, [0, 2, 3, 1])
        input_shape = x.get_shape().as_list()
        num_inputs = 1
        for dim in input_shape[1:]:
            num_inputs *= dim
        return tf.reshape(x, [-1, num_inputs], name='flatten')

    def residual2d(self, inputs, network, units=None, scale=1.0, activate=False):
        outputs = network(inputs)
        c_axis = -1 if self.data_format == 'channels_last' else 1
        h_axis =  1 if self.data_format == 'channels_last' else 2
        w_axis = h_axis + 1
        ishape, oshape = [y.get_shape().as_list() for y in [inputs, outputs]]
        ichans, ochans = ishape[c_axis], oshape[c_axis]
        strides = ((ishape[h_axis] - 1) // oshape[h_axis] + 1,
                   (ishape[w_axis] - 1) // oshape[w_axis] + 1)
        with tf.name_scope('residual'):
            if (ochans != ichans or strides[0] != 1 or strides[1] != 1):
                inputs = self.conv2d_linear(inputs, units, 1, strides, 'SAME')
            x = inputs + scale * outputs
            if activate:
                x = self.activate(x)
        return x

    def dropout(self, inputs, *args, **kwargs):
        return tf.layers.dropout(inputs, training=self.training, *args, **kwargs)

    def inception_module(self, inputs, name, cols):
        with tf.name_scope(name):
            col_layers      = []
            col_layer_sizes = []
            for c, col in enumerate(cols):
                col_layers.append([])
                col_layer_sizes.append([])
                x = inputs
                for l, layer in enumerate(col):
                    ltype, args = layer[0], layer[1:]
                    if   ltype == 'conv2d_linear': x = self.conv2d_linear(x, *args)
                    elif ltype == 'conv2d':        x = self.conv2d(x, *args)
                    elif ltype == 'mpool2d':       x = self.max_pooling2d(x, *args)
                    elif ltype == 'apool2d':       x = self.average_pooling2d(x, *args)
                    elif ltype == 'share':
                        # Share matching layer from previous column
                        x = col_layers[c-1][l]
                    else: raise KeyError("Invalid layer type for " +
                                         "inception module: '%s'" % ltype)
                    col_layers[c].append(x)
            catdim  = -1 if self.data_format == 'channels_last' else 1
            catvals = [layers[-1] for layers in col_layers]
            x = tf.concat(catvals, catdim)
            return x

