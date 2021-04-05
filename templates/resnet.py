
import cfg_base as cfgb
import layers as xlayers
import blocks as xblocks
import tensorflow as tf
from copy import copy



def pre_build(ids, trainable = True):
	"""default ResNet 34 or 50
	"""
	features = 64 # root features of the first block, it will be doubled for each block
	nsublayers = 3
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
	nlayers = [3, 4, 6, 3]
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
	

