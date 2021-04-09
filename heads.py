

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import layers as xlayer

class classifier_CAM_2D(tf.keras.layers.Layer):
    """ header of classification activation map
        implemented by a global average pooling 
        for multi-labeled classification problem, activation should be `sigmoid`
    """
    def __init__(self, nclass, activation = 'softmax', **kwarg):
        super(classifier_CAM_2D, self).__init__(**kwarg)
        self.nclass = nclass
        self.activation = activation
        self.layer_average = tf.keras.layers.GlobalAveragePooling2D()
        self.layer_dense = tf.keras.layers.Dense(units = nclass, use_bias = False, activation=self.activation)
        self.layer_heatmap = tf.keras.layers.Conv2D(filters = nclass, 
                kernel_size=1, strides=1, use_bias = False, trainable = False)

    def __call__(self, inputs):
        x = self.layer_average(inputs)
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
        config=super(classifier_CAM_2D, self).get_config()
        config.update({ 'nclass':self.nclass, 'activation':self.activation})
        return config


class RPN2D_box_classifier(tf.keras.layers.Layer):
    def __init__(self, nclass, anchors_per_location = 9, nlayers = 4, nfilters =256, **kwarg):
        super(RPN2D_box_classifier, self).__init__(**kwarg)
        self.nlayers = nlayers
        self.nfilters = nfilters
        self.anchors_per_location = anchors_per_location
        self.nclass = nclass

    def build(self):
        self.fcn = []
        self.norm_layer = []
        for i in range(self.nlayers):
            self.fcn.append(tf.keras.layers.Conv2D(nfilters = self.nfilters,
                                   kernel_size = (3,3),
                                   padding = 'same',
                                   bias_initializer=tf.zeros_initializer(),
                                   activation = None,
                                   kernel_initializer=
                                        tf.keras.initializers.RandomNormal(stddev=0.01),
                                   name = 'PRN2D_box-FCN-cls-{lv}'.format(lv=i)))
            self.norm_layer.append(xlayer.batch_norm_activation(
                                    name = 'RPN2D-box-BNA-cls-{lv}'.format(lv=i)))
        # nclass + 1 , 1 stands for no objects
        self.pred = tf.keras.layers.Conv2D((self.nclass+1)*self.anchor_per_location,
                                            kernel_size=(3,3),
                                            bias_initializer=tf.zeros_initializer(),
                                            kernel_initializer=
                                                    tf.keras.initializers.RandomNormal(stddev=1e-5),
                                            padding='same',
                                            name = 'PRN2D-box-cls-predicit')

    def __cal__(self, inputs):
        with tf.name_scope('RPN2D_box_classifer'):
            x = inputs
            for i in range(self.nlayers):
                x = self.fcn[i](x)
                x = self.norm_layer[i](x)
            x = self.pred(x)
        return x

    def mlv_feature_classifier(self, features):
        outputs = []
        for feature in features:
            outputs.append( self.__call__(feature) )
        return outputs

    def get_config(self):
        config=super(RPN2D_box_classifier, self).get_config()
        config.update({ 'nclass':self.nclass,
                        'anchors_per_location':self.anchors_per_location,
                        'nlayer':self.nlayers,
                        'nfilters':self.nfilters})
        return config

class RPN2D_box_regressor(tf.keras.layers.Layer):
    def __init__(self, anchors_per_location = 9, nlayers = 4, nfilters =256, **kwarg):
        super(RPN2D_box_regressor, self).__init__(**kwarg)
        self.nlayers = nlayers
        self.nfilters = nfilters
        self.anchors_per_location = anchors_per_location

    def build(self):
        self.fcn = []
        self.norm_layer = []
        for i in range(self.nlayers):
            self.fcn.append(tf.keras.layers.Conv2D(nfilters = self.nfilters,
                                   kernel_size = (3,3),
                                   padding = 'same',
                                   bias_initializer=tf.zeros_initializer(),
                                   activation = None,
                                   kernel_initializer=
                                        tf.keras.initializers.RandomNormal(stddev=0.01),
                                   name = 'PRN2D_box-FCN_reg-{lv}'.format(lv=i)))
            self.norm_layer.append(xlayer.batch_norm_activation(
                                    name = 'RPN2D-box-BNA-reg-{lv}'.format(lv=i)))
        self.pred = tf.keras.layers.Conv2D(4*self.anchor_per_location,
                                            kernel_size=(3,3),
                                            bias_initializer=tf.zeros_initializer(),
                                            kernel_initializer=
                                                    tf.keras.initializers.RandomNormal(stddev=1e-5),
                                            padding='same',
                                            name = 'PRN2D-box-reg-predicit')

    def __cal__(self, inputs):
        with tf.name_scope('RPN2D_box_regressor'):
            x = inputs
            for i in range(self.nlayers):
                x = self.fcn[i](x)
                x = self.norm_layer[i](x)
            x = self.pred(x)
        return x

    def mlv_feature_regressor(self, features):
        outputs = []
        for feature in features:
            outputs.append( self.__call__(feature) )
        return outputs

    def get_config(self):
        config=super(RPN2D_box_regressor, self).get_config()
        config.update({ 'anchors_per_location':self.anchors_per_location,
                        'nlayer':self.nlayers,
                        'nfilters':self.nfilters})
        return config
