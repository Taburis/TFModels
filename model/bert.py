
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from config import ParaSet as pset
import module.Transformer as xtr
#from module import Transfomer as xtr
import unicodedata
import six
import re

"""
a text tokenizer from tensorflow is used
"""

def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")

class tokenizer(object):
    def __init__(self, uncased = False):
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(
            lower= uncased)
        self.is_init = False

    def initializer(self, text):
        text = convert_to_unicode(text)
        self.tokenizer.fit_on_texts()
        self.is_init = True

    def __call__(self, text):
        x = self.tokenizer.texts_to_sequences(text)
        x = tf.keras.preprocessing.sequence.pad_sequences(x,padding='post')
        mask = self.padding_mask(x)
        return x, mask

    def padding_mask(self,x):
        x = tf.cast(tf.math.equal(x, 0), tf.float32)
        return x

    def looking_ahead_mask(self,size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (seq_len, seq_len)

class model(tf.keras.Model):
    """
        the encoder cfg is a dict with option:
        L: number of layeres
        H: model depth
        A: number of attension heads
        dff: depth of hidden FF net, default will be 1024
        dropout_rate: drop rate, 0.1 as default
        input_vocab_size: the input vocabulary size, get from the tokenizer
        maximum_position_encoding
        positional_encoding: object for encoding the position information.
                         The default sinusoid_position_encoder will be used
                         if this part is missing.
    """
    def __init__(encoder_cfg, **kwargs):
        super(model, self).__init__(**kwargs)
        self.cfg = bert_cfg
        self.encoder_cfg = bert_cfg.clone()
        self.encoder_cfg['d_model'] = self.encoder_cfg.pop('H')
        self.encoder_cfg['num_heads'] = self.encoder_cfg.pop('A')
        self.encoder_cfg['num_layers'] = self.encoder_cfg.pop('L')
        if not self.encoder['positional_encoding']: 
            self.encoder_cfg['positional_encoding'] = xtr.sinusoid_position_encoder_generator(
                    self.encoder_cfg['maximum_position_encoding'])
   
    def build(self):
        xtr.Encoder(**encoder_cfg)

       
