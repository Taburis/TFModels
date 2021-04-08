
"""
Template for U-net 

based on the paper https://arxiv.org/abs/1505.04597

pre_build will generate a default cfg files consumed by build_template.
Adjust the configuration `cfg` and the function
	build_template(cfg)
returns the decorated resnet callable function.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cfg_base as cfgb
import layers as xlayers
import blocks as xblocks
import tensorflow as tf
from copy import copy

class module_unet(tf.keras.layers.Layer):
    def __init__(self, root_feature = 64, nsublayer_conv2d = 2, nlevel = 4, 
                 keep_multi_level_features= False, **kwarg):
        """default U-net 
            nsublayers: how many conv2d in each depth
        """
        super(module_unet, self).__init__(**kwarg)
        self.root_feature = root_feature
        self.nlevel = nlevel
        self.nsublayer_conv2d = nsublayer_conv2d

        features = root_feature # root features of the first block, it will be doubled for each block
        self.keep_multi_level_features = keep_multi_level_features
        self.multi_level_features = {}
    
        default_cfg_conv2d = {'filters':64,
                          'kernel_size':3,
                          'strides':1,
                          'padding':'same',
                          'activation':'relu',
                          'use_bias':True,
                          'kernel_initializer':tf.initializers.VarianceScaling()}
    
        default_cfg_block = cfgb.cfg_base(
                        conv2d = copy(default_cfg_conv2d),
                        nlayer = nsublayer_conv2d,
                        residual_learning = True)
    
        cfg_down = []
        cfg_up   = []
        for i in range(nlevel):
            cfu = default_cfg_block.clone()
            cfu.conv2d['filters'] = features*pow(2,i)
            cfd = default_cfg_block.clone()
            cfd.conv2d['filters'] = features*pow(2,i)
            cfg_down.append(cfd)
            cfg_up.append(cfu)
        
        cfg_top = default_cfg_block.clone()
        cfg_top.conv2d['filters'] = features*pow(2,nlevel)
        
        self.cfg_unet = cfgb.cfg_base( up_stream = cfg_up, down_stream=cfg_down,
                    cfg_top = cfg_top, root_feature = self.root_feature, nlevel = nlevel,
                    drop_rate = 0.1)
       
    def build(self):
        """ 2D residual network template
        refered to the paper: https://arxiv.org/pdf/1512.03385.pdf
        34-layer default	
        """
        ports = {}
        layer_up= {}
        layer_down= {}
        
        layer_top = xblocks.sequential_conv2d(self.cfg_unet.cfg_top)
        cfg_up = self.cfg_unet.up_stream
        cfg_down = self.cfg_unet.down_stream
        nlevel = self.cfg_unet.nlevel
        root_feature = self.cfg_unet.root_feature
        drop_rate = self.cfg_unet.drop_rate
        
        for i in range(len(self.cfg_unet.up_stream)):
            layer_up[i]   = xblocks.sequential_conv2d(cfg_up[i], trainable)
            layer_down[i] = xblocks.sequential_conv2d(cfg_down[i], trainable)

    def mlv_features(self):
        return self.multi_level_features
       	
    def __call__(self, inputs):
        x = inputs
        uptensor = {}
        for i in range(nlevel):
            x = layer_up[i](x)
            if drop_rate > 0:
            	x = tf.keras.layers.Dropout(rate=drop_rate)(x)
            uptensor[i] = x
            x = tf.keras.layers.MaxPool2D()(x)
        x = layer_top(x)
        for i in range(nlevel):
            x = tf.keras.layers.Conv2DTranspose(filters=root_feature*pow(2,nlevel-i-1), 
                                            kernel_size = 2, strides = 2)(x)
            x = tf.concat([uptensor[nlevel-i-1],x], axis=-1)
            x = layer_down[nlevel - i-1](x)
            if self.keep_multi_level_features:
                self.multi_level_features[i] = tf.Identity(x, 
                        name =self.name +'_mlf_{LV}'.format(LV=i))
        return x
    
    def get_config(self):
        config = super(module_unet, self).get_config()
        config['root_feature'] = self.root_feature
        config['nlevel'] = self.nlevel
        config['nsublayer_conv2d'] = self.nsublayer_conv2d
        config['keep_multi_level_features'] = self.keep_multi_level_features

