

"""
Template for Residual Networks

based on the paper https://arxiv.org/pdf/1512.03385.pdf

pre_build will generate a default cfg files consumed by build_template.
	cfg = pre_build('34') or '50'
Adjust the configuration `cfg` and the function
	build_template(cfg)
returns the decorated resnet callable function.

notice, the normalization layer is not added for residual shortcut
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cfg_base as cfgb
import layers as xlayers
import blocks as xblocks
import tensorflow as tf
from copy import copy



def pre_build(ids, trainable = True):
    """default ResNet 34 or 50
    """
    features = 64 # root features of the first block, it will be doubled for each block
    nsublayers = 3 # usually the structure that shortcut 3 sublayers for residual is called
    			   # bottle neck. But resnet with depth 34, no bottle neck is used. 
    if ids == '34' : 
    	nsublayers = 2
    default_cfg_conv2d = {'filters':64,
                          'kernel_size':3,
                          'strides':1,
                          'padding':'same',
                          'activation':'relu',
                          'use_bias':False,
                          'kernel_initializer':tf.initializers.VarianceScaling()}
    
    default_cfg_block = cfgb.cfg_base(
                        conv2d = copy(default_cfg_conv2d),
                        nlayer = 2,
                        residual_learning = True,
                        trainable = trainable)
    
    cfg_bottle_neck = default_cfg_block.clone(nlayer=nsublayers)
    cfg_bn_1 = cfg_bottle_neck.clone()
    cfg_bn_1.conv2d['filters'] = features
    cfg_bn_2 = cfg_bottle_neck.clone()
    cfg_bn_2.conv2d['filters'] = features*2
    cfg_bn_3 = cfg_bottle_neck.clone()
    cfg_bn_3.conv2d['filters'] = features*4
    cfg_bn_4 = cfg_bottle_neck.clone()
    cfg_bn_4.conv2d['filters'] = features*8
    cfg = [cfg_bn_1, cfg_bn_2,cfg_bn_3,cfg_bn_4]
    nlayers = [3, 4, 6, 3] # the numbers of shortcuts in 34 and 50 resnet are the same 
    return cfg, nlayers

def build_template(cfg_resnet, trainable=True):
    """ 2D residual network template
    refered to the paper: https://arxiv.org/pdf/1512.03385.pdf
    34-layer default	
    """
    ports = {}
    cfg, nly = cfg_resnet
    for i in range(len(cfg)):
        ports[i+1] = xblocks.sequential_block(nlayer = nly[i], 
                                              module = xblocks.sequential_conv2d, 
                                              cfg = cfg[i],
                                              trainable = trainable)
    def resnet_imp(inputs, trainable = trainable):
        x = inputs
        for key, layer in ports.items():
            x = layer(x)
        return x
    return ports, resnet_imp
	

