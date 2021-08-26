

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from config import ParaSet as pset
import module.Transformer as xtr
import tensorflow as tf
#from module import Transfomer as xtr
import unicodedata
import six
import re

"""
Transformer Encoder-based feature extractor + SV Regressor

For doing time series analyzing, removed the embedding layer
"""

def const_mask_processor(x):
    m = tf.ones()
    return x, 

class config(pset):
    """
        the encoder cfg is a dict with option:
        L: number of layeres
        H: model depth
        A: number of attension heads
        dff: depth of hidden FF net
        dropout_rate: drop rate, 0.1 as default
        positional_encoding: object for encoding the position information.
                         The default sinusoid_position_encoder will be used
                         if this part is missing.

        nfeatures: The number of cell in Dense layer for regressor
        nchannel:  output channels
    """

    def __init__(self,
                 #input_size,
                 L= 4, H = 1024, A= 8,
                 dff = 512,
                 dropout_rate = 0.1,
                 maximum_position_encoding = 650,
                 nfeatures = 128,
                 nchannel = 112
                 ):
        super(config, self).__init__()
        self.encoder_cfg = pset(num_layers = L, d_model = H, num_heads = A, 
                                dff = dff,
                                dropout_rate = dropout_rate,
                                positional_encoding = xtr.sinusoid_position_encoder_generator(
                                maximum_position_encoding),
                                maximum_position_encoding = maximum_position_encoding,
                               )
        self.nfeatures=nfeatures
        self.nchannel=nchannel

class model(tf.keras.Model):
    def __init__(self, bert_cfg,  training = True, **kwargs):
        super(model, self).__init__(**kwargs)
        self.cfg = bert_cfg
        self.encoder_cfg = bert_cfg.encoder_cfg
#        self.preprocessor = preprocessor

        self.encoder = xtr.Encoder(**(self.encoder_cfg))
        self.feature_nn = tf.keras.layers.Dense(self.cfg.nfeatures, activation='relu')
        self.regressor = tf.keras.layers.Dense(1, use_bias = False)
    def padding_mask(x):
        return tf.cast(tf.math.equal(x, -99), tf.float32)

    def build_model(self):
        """ return a keras model object"""
        inputs = tf.keras.layers.Input(shape=(), dtype=tf.float32, name='float32_input')
        m = self.padding_mask(inputs)
#        x,m = self.preprocessor(inputs)
        code = self.encoder(x, training, m)
        features = self.feature_nn(code)
        output = self.regressor(features)
        return tf.Model(inputs, output)

