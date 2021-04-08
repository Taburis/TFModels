
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cfg_base as cfgb
import layers as xlayers
import blocks as xblocks
import tensorflow as tf
import architecture.unet as unet
import heads as xheads 
from copy import copy

class hpa_unet(object):
    def __init__(self, name = 'hpa_unet', input_shape = (1024, 1024, 4)):
        self.name = name
        self.input_shape = input_shape
        self.feature_extractor = unet.module_unet(root_feature = 32, keep_multi_level_features = False)
        self.feature_extractor.build()
        self.head_classifier = xheads.classifier_CAM_2D(nclass = 20)
        self.segmentation = self.keras.layers.Conv2D(filters = 2, kernel_size = (3,3),
                                                     strides = 1,
                                                     padding = 'same',
                                                     use_bias = False,
                                                     name = 'hpa-unet-seg-head')

    def build(self, trainable = True):
        inputs = tf.Input(shape = self.input_shape)
        x = self.feature_extractor(inputs)
        seg_pred = self.segmentation(x)
        x = tf.stack([seg_pred, inputs[...,3]])
        cls_pred = self.head_classifier(x)
        outputs = [cls_pred, seg_pred]
        self.nn = tf.keras.Model(inputs = x, outputs = outputs, name=self.name, trainable=True)

    def __call__(self, inputs):
        return self.nn(inputs)

     
