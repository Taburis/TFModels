
"""
collection of customed layers
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class residual2D(tf.keras.layers.Layer):
    """ Known as shortcut in residual learning. 
    """
    def __init__(self, 
                 use_projection = False,
                 trainable = None, **kwargs):
        super(residual2D, self).__init__(self, **kwargs)
        self.use_projection = use_projection
        self.trainable = trainable
        self.project_layer = tf.keras.layers.Conv2D(
            filters = nfilter, 
            kernel_size = 1, 
            strides = 1, 
            padding = 'valid', 
            trainable = self.trainable)
        
    def call(self, x, r, dim = 3):
        """
        Args: 
        x: the shortcut from inputs
        r: residual inputs
        dim: which dimension is the filters
        """
        shortcut = x
        nfilter = r.shape[dim]
        if not r.shape[dim] == x.shape[dim] : 
            self.use_projection = True
        if self.use_projection : 
            shortcut = self.project_layer(x)
#   print('filters for shortcut: ', nfilter, shortcut.shape[dim])
        return shortcut + r
 
    def get_config(self):
        return {'use_projection':self.use_projection, 'trainable':self.trainable}


class batch_norm_activation(tf.keras.layers.Layer):
    def __init__(self,
               momentum=0.997,
               epsilon=1e-4,
               trainable=True,
               init_zero=False,
               use_activation=True,
               activation='relu',
               fused=True,
               name=None):
        """A class to construct layers for a batch normalization followed by a ReLU.
        Args:
          momentum: momentum for the moving average.
          epsilon: small float added to variance to avoid dividing by zero.
          trainable: `bool`, if True also add variables to the graph collection
            GraphKeys.TRAINABLE_VARIABLES. If False, freeze batch normalization
            layer.
          init_zero: `bool` if True, initializes scale parameter of batch
            normalization with 0. If False, initialize it with 1.
          fused: `bool` fused option in batch normalziation.
          use_actiation: `bool`, whether to add the optional activation layer after
            the batch normalization layer.
          activation: 'string', the type of the activation layer. Currently support
            `relu` and `swish`.
          name: `str` name for the operation.
        """
        super(batch_norm_activation, self).__init__(trainable=trainable, **kwarg)
        if init_zero:
            gamma_initializer = tf.keras.initializers.Zeros()
        else:
            gamma_initializer = tf.keras.initializers.Ones()
        self._normalization_op = tf.keras.layers.BatchNormalization(
            momentum=momentum,
            epsilon=epsilon,
            center=True,
            scale=True,
            trainable=trainable,
            fused=fused,
            gamma_initializer=gamma_initializer,
            name=name)
        self._use_activation = use_activation
        if activation == 'relu':
            self._activation_op = tf.nn.relu
        elif activation == 'swish':
            self._activation_op = tf.nn.swish
        else:
            raise ValueError('Unsupported activation `{}`.'.format(activation))

    def __call__(self, inputs, is_training=None):
        """Builds the normalization layer followed by an optional activation layer.
        Args:
          inputs: `Tensor` of shape `[batch, channels, ...]`.
          is_training: `boolean`, if True if model is in training mode.
        Returns:
          A normalized `Tensor` with the same `data_format`.
        """
        # We will need to keep training=None by default, so that it can be inherit
        # from keras.Model.training
        if is_training and self.trainable:
          is_training = True
        inputs = self._normalization_op(inputs, training=is_training)
        
        if self._use_activation:
          inputs = self._activation_op(inputs)
        return inputs

