import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa


# helper function for creating a convolutional layer in a TensorFlow mode
def conv(inputs,
         nfilters,  # number of filters to use in the layer
         ksize,  # size of the filters (kernel size)
         stride=1,  # stride to use for the convolution (default is 1)
         padding='SAME',  # padding to use for the convolution
         use_bias=True,
         activation_fn=tf.compat.v1.nn.relu,  # the activation function to use (default is ReLU)
         initializer=tf.keras.initializers.VarianceScaling,  # the weight initializer to use
         regularizer=None,  # the regularization function to use
         scope=None,  # the variable scope to use for the layer
         reuse=None):  # whether to reuse the layer's variables
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        n_in = inputs.get_shape().as_list()[-1]
        weights = tf.compat.v1.get_variable(
            'weights',
            shape=[ksize, ksize, n_in, nfilters],
            dtype=inputs.dtype.base_dtype,
            initializer=initializer,
            collections=[tf.compat.v1.GraphKeys.WEIGHTS, tf.compat.v1.GraphKeys.VARIABLES],
            regularizer=regularizer)

        strides = [1, stride, stride, 1]
        current_layer = tf.compat.v1.nn.conv2d(inputs, weights, strides, padding=padding)

        if use_bias:
            biases = tf.compat.v1.get_variable(
                'biases',
                shape=[nfilters, ],
                dtype=inputs.dtype.base_dtype,
                initializer=tf.compat.v1.constant_initializer(0.0),
                collections=[tf.compat.v1.GraphKeys.BIASES, tf.compat.v1.GraphKeys.VARIABLES])
            current_layer = tf.compat.v1.nn.bias_add(current_layer, biases)

        if activation_fn is not None:
            current_layer = activation_fn(current_layer)

        return current_layer


# fractionally strided convolution or deconvolution. Defines a transposed convolutional layer in a neural network model
def transpose_conv(inputs,
                   nfilters,
                   ksize,
                   stride=1,
                   padding='SAME',
                   use_bias=True,
                   activation_fn=tf.compat.v1.nn.relu,
                   initializer=tf.keras.initializers.VarianceScaling,
                   regularizer=None,
                   scope=None,
                   reuse=None):
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        n_in = inputs.get_shape().as_list()[-1]
        weights = tf.compat.v1.get_variable(
            'weights',
            shape=[ksize, ksize, n_in, nfilters],
            dtype=inputs.dtype.base_dtype,
            initializer=initializer,
            collections=[tf.compat.v1.GraphKeys.WEIGHTS, tf.compat.v1.GraphKeys.VARIABLES],
            regularizer=regularizer)

        bs, h, w, c = inputs.get_shape().as_list()
        strides = [1, stride, stride, 1]
        out_shape = [bs, stride * h, stride * w, c]
        current_layer = tf.compat.v1.nn.conv2d_transpose(inputs, weights, out_shape, strides, padding=padding)

        if use_bias:
            biases = tf.compat.v1.get_variable(
                'biases',
                shape=[nfilters, ],
                dtype=inputs.dtype.base_dtype,
                initializer=tf.compat.v1.constant_initializer(0.0),
                collections=[tf.compat.v1.GraphKeys.BIASES, tf.compat.v1.GraphKeys.VARIABLES])
            current_layer = tf.compat.v1.nn.bias_add(current_layer, biases)

        if activation_fn is not None:
            current_layer = activation_fn(current_layer)

        return current_layer


# performing 1D convolution on the input tensor
def conv1(inputs,
          nfilters,
          ksize,
          stride=1,
          padding='SAME',
          use_bias=True,
          activation_fn=tf.compat.v1.nn.relu,
          initializer=tf.keras.initializers.VarianceScaling(),
          regularizer=None,
          scope=None,
          reuse=None):
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        n_in = inputs.get_shape().as_list()[-1]
        weights = tf.compat.v1.get_variable(
            'weights',
            shape=[ksize, n_in, nfilters],
            dtype=inputs.dtype.base_dtype,
            initializer=initializer,
            collections=[tf.compat.v1.GraphKeys.WEIGHTS, tf.compat.v1.GraphKeys.VARIABLES],
            regularizer=regularizer)

        current_layer = tf.compat.v1.nn.conv1d(inputs, weights, stride, padding=padding)

        if use_bias:
            biases = tf.compat.v1.get_variable(
                'biases',
                shape=[nfilters, ],
                dtype=inputs.dtype.base_dtype,
                initializer=tf.compat.v1.constant_initializer(0.0),
                collections=[tf.compat.v1.GraphKeys.BIASES, tf.compat.v1.GraphKeys.VARIABLES])
            current_layer = tf.compat.v1.nn.bias_add(current_layer, biases)

        if activation_fn is not None:
            current_layer = activation_fn(current_layer)

        return current_layer


# performs 1-dimensional atrous convolution. Atrous convolution, also known as dilated convolution, is a type of
# convolution operation that applies a filter to the input with gaps or dilations between the filter elements.
def atrous_conv1d(inputs,
                  nfilters,
                  ksize,
                  rate=1,
                  padding='SAME',
                  use_bias=True,
                  activation_fn=tf.compat.v1.nn.relu,
                  initializer=tf.keras.initializers.VarianceScaling(),
                  regularizer=None,
                  scope=None,
                  reuse=None):
    """ Use tf.compat.v1.nn.atrous_conv2d and adapt to 1d"""
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        # from (bs,width,c) to (bs,width,1,c)
        inputs = tf.compat.v1.expand_dims(inputs, 2)

        n_in = inputs.get_shape().as_list()[-1]
        weights = tf.compat.v1.get_variable(
            'weights',
            shape=[ksize, 1, n_in, nfilters],
            dtype=inputs.dtype.base_dtype,
            initializer=initializer,
            collections=[tf.compat.v1.GraphKeys.WEIGHTS, tf.compat.v1.GraphKeys.VARIABLES],
            regularizer=regularizer)

        current_layer = tf.compat.v1.nn.atrous_conv2d(inputs, weights, rate, padding=padding)

        # Resize into (bs,width,c)
        current_layer = tf.compat.v1.squeeze(current_layer, squeeze_dims=[2])

        if use_bias:
            biases = tf.compat.v1.get_variable(
                'biases',
                shape=[nfilters, ],
                dtype=inputs.dtype.base_dtype,
                initializer=tf.compat.v1.constant_initializer(0.0),
                collections=[tf.compat.v1.GraphKeys.BIASES, tf.compat.v1.GraphKeys.VARIABLES])
            current_layer = tf.compat.v1.nn.bias_add(current_layer, biases)

        if activation_fn is not None:
            current_layer = activation_fn(current_layer)

        return current_layer


#  applies 3D convolution to the input tensor
def conv3(inputs,
          nfilters,
          ksize,
          stride=1,
          padding='SAME',
          use_bias=True,
          activation_fn=tf.compat.v1.nn.relu,
          initializer=tf.keras.initializers.VarianceScaling(),
          regularizer=None,
          scope=None,
          reuse=None):
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        n_in = inputs.get_shape().as_list()[-1]
        weights = tf.compat.v1.get_variable(
            'weights',
            shape=[ksize, ksize, ksize, n_in, nfilters],
            dtype=inputs.dtype.base_dtype,
            initializer=initializer,
            collections=[tf.compat.v1.GraphKeys.WEIGHTS, tf.compat.v1.GraphKeys.VARIABLES],
            regularizer=regularizer)

        strides = [1, stride, stride, stride, 1]
        current_layer = tf.compat.v1.nn.conv3d(inputs, weights, strides, padding=padding)

        if use_bias:
            biases = tf.compat.v1.get_variable(
                'biases',
                shape=[nfilters, ],
                dtype=inputs.dtype.base_dtype,
                initializer=tf.compat.v1.constant_initializer(0.0),
                collections=[tf.compat.v1.GraphKeys.BIASES, tf.compat.v1.GraphKeys.VARIABLES])
            current_layer = tf.compat.v1.nn.bias_add(current_layer, biases)

        if activation_fn is not None:
            current_layer = activation_fn(current_layer)

        return current_layer


# defines a fully connected layer in a neural network. A fully connected layer, also known as a dense layer,
# connects each neuron in the previous layer to every neuron in the current layer
def fc(inputs, nfilters, use_bias=True, activation_fn=tf.compat.v1.nn.relu,
       initializer=tf.keras.initializers.VarianceScaling,
       regularizer=None, scope=None, reuse=None):
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        n_in = inputs.get_shape().as_list()[-1]
        weights = tf.compat.v1.get_variable(
            'weights',
            shape=[n_in, nfilters],
            dtype=inputs.dtype.base_dtype,
            initializer=initializer,
            regularizer=regularizer)

        current_layer = tf.compat.v1.matmul(inputs, weights)

        if use_bias:
            biases = tf.compat.v1.get_variable(
                'biases',
                shape=[nfilters, ],
                dtype=inputs.dtype.base_dtype,
                initializer=tf.compat.v1.constant_initializer(0))
            current_layer = tf.compat.v1.nn.bias_add(current_layer, biases)

        if activation_fn is not None:
            current_layer = activation_fn(current_layer)

    return current_layer


# implements batch normalization, which is a technique used to improve the training stability and performance of deep
# neural networks.
def batch_norm(inputs, center=False, scale=False,
               decay=0.999, epsilon=0.001, reuse=None,
               scope=None, is_training=False):
    return tf.compat.v1.contrib.layers.batch_norm(
        inputs, center=center, scale=scale,
        decay=decay, epsilon=epsilon, activation_fn=None,
        reuse=reuse, trainable=False, scope=scope, is_training=is_training)


relu = tf.compat.v1.nn.relu


#  crops a given inputs tensor to have the same shape as another tensor "like".
def crop_like(inputs, like, name=None):
    with tf.compat.v1.name_scope(name):
        _, h, w, _ = inputs.get_shape().as_list()
        _, new_h, new_w, _ = like.get_shape().as_list()
        crop_h = (h - new_h) / 2
        crop_w = (w - new_w) / 2
        cropped = inputs[:, crop_h:crop_h + new_h, crop_w:crop_w + new_w, :]
        return cropped
