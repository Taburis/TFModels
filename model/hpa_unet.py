from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cfg_base as cfgb
import tensorflow as tf
from module import Unet as unet
from module import RetinaNet as rn
from copy import copy

class hpa_unet(tf.keras.Model):
    def __init__(self, name = 'hpa_unet', cls_activation = 'sigmoid', seg_activation = 'softmax', input_shape = (1024, 1024, 4), **kwargs):
        super(hpa_unet, self).__init__(**kwargs)
        self.feature_extractor = unet.unet(root_feature = 32, keep_multi_level_features = True)
        self.feature_extractor.build()
#        self.segmentation = tf.keras.layers.Conv2D(filters = 1, 
#                                                   kernel_size = (3,3),
#                                                   strides = 1,
#                                                   padding = 'same',
#                                                   use_bias = True,
#                                                   activation = seg_activation,
#                                                   name = 'hpa-unet-seg-head')
        inputs = tf.keras.Input(shape = input_shape, dtype= 'float32')
        x = self.feature_extractor(inputs)
        feature_size = [v.shape[1] for k,v in self.feature_extractor.multi_level_features.items()][0:-1]
        features = [v for v in self.feature_extractor.multi_level_features.values()][0:-1]
        self.anchors = rn.Anchor(input_size = input_shape[0], 
                                     base_scale = 9*9,
                                     feature_size = feature_size)
        self.make_training_labels = self.anchors.generate_labels
        self.head = rn.Head(name = 'hpa-unet-cls-head', 
                                nclass = 1,
                                anchor_size = self.anchors.size)

        self.feature_alignment_layer = [] # these layers make the fp has the same length of channels
        self.fp = []
        for v in features:
            layer = tf.keras.layers.Conv2D(filters = 512, kernel_size=1, padding = 'same', strides = 1,
            use_bias = False, activation = 'relu')
            self.feature_alignment_layer.append(layer)
            self.fp.append(layer(v))
        res = self.head(self.fp)
        self.nn = tf.keras.Model(inputs = inputs, outputs = res, name=name, trainable=True)

    def __call__(self, inputs):
        return self.nn(inputs)
