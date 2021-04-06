

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



def pre_build(root_features = 32, nsublayer = 2, nstage = 4, trainable = True):
    """default U-net 
        nsublayers: how many conv2d in each depth
    """
    features = root_features # root features of the first block, it will be doubled for each block
    
    default_cfg_conv2d = {'filters':64,
                          'kernel_size':3,
                          'strides':1,
                          'padding':'same',
                          'activation':'relu',
                          'use_bias':True,
                          'kernel_initializer':tf.initializers.VarianceScaling()}
    
    default_cfg_block = cfgb.cfg_base(
                        conv2d = copy(default_cfg_conv2d),
                        nlayer = 2,
                        residual_learning = True,
                        trainable = trainable)
    
    cfg_down = []
    cfg_up   = []
    for i in range(nstage):
        cfu = default_cfg_block.clone()
        cfu.conv2d['filters'] = features*pow(2,i)
        cfd = default_cfg_block.clone()
        cfd.conv2d['filters'] = features*pow(2,i)
        cfg_down.append(cfd)
        cfg_up.append(cfu)
    
    cfg_top = default_cfg_block.clone()
    cfg_top.conv2d['filters'] = features*pow(2,nstage)
    
    cfg_unet = cfgb.cfg_base( up_stream = cfg_up, down_stream=cfg_down,
                cfg_top = cfg_top, root_features = root_features, nstage = nstage,
                drop_rate = 0.1)
    
    return cfg_unet 

def build_template(cfg_unet,  trainable=True):
    """ 2D residual network template
    refered to the paper: https://arxiv.org/pdf/1512.03385.pdf
    34-layer default	
    """
    ports = {}
    layer_up= {}
    layer_down= {}
    
    layer_top = xblocks.sequential_conv2d(cfg_unet.cfg_top, trainable)
    cfg_up = cfg_unet.up_stream
    cfg_down = cfg_unet.down_stream
    nstage = cfg_unet.nstage
    root_features = cfg_unet.root_features
    drop_rate = cfg_unet.drop_rate
    
    for i in range(len(cfg_unet.up_stream)):
        layer_up[i]   = xblocks.sequential_conv2d(cfg_up[i], trainable)
        layer_down[i] = xblocks.sequential_conv2d(cfg_down[i], trainable)
    	
    def nn_imp(inputs, trainable = trainable):
        x = inputs
        uptensor = {}
        for i in range(nstage):
            x = layer_up[i](x)
            if drop_rate > 0:
            	x = tf.keras.layers.Dropout(rate=drop_rate)(x)
            uptensor[i] = x
            x = tf.keras.layers.MaxPool2D()(x)
        x = layer_top(x)
        for i in range(nstage):
            x = tf.keras.layers.Conv2DTranspose(filters=root_features*pow(2,nstage-i-1), kernel_size = 2, strides = 2)(x)
            x = tf.concat([uptensor[nstage-i-1],x], axis=-1)
            x = layer_down[nstage - i-1](x)
        return x
    
    return nn_imp
	

