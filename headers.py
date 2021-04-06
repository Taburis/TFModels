

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

class header_classifer_CAM_2D(tf.keras.layers.Layer):
	""" header of classification activation map
        implemented by a global average pooling 
        for multi-labeled classification problem, activation should be `sigmoid`
    """
    def __init__(self, nclass, activation = 'softmax', **kwarg):
        super(header_classifer_CAM_2D, self).__init__(**kwarg):
        self.trainable = trainable
        self.nclass = nclass
        self.actiavtion = activation
        self.layer_average = tf.keras.layers.GlobalAveragePooling2D()
        self.layer_dense = tf.keras.Dense(units = nclass, use_bias = False, activation=self.activation)
        self.layer_heatmap = tf.keras.layers.Conv2D(nfilters = ncalss, 
                kernel_size=1, strides=1, use_bias = False, trainable = False)

    def call(self, inputs):
        x = self.layer_averge(inputs)
        x = self.layer_dense(x)
        return x

    def load_heatmap(self):
        weights = self.layer_dense.get_weights()
        weights = [[weights]]
        np.swapaxes(weights, 0, 2)
        self.layer_heatmap.set_weights(weights)
        def heatmap(inputs):
            return self.layer_heatmap(inputs)
        return heatmap

    def get_config(self):
        config=super(header_classifer_CAM_2D, self).get_config()
        config.update({ 'nclass':nclass, 'activation':activation})
        return config
