
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

def text_preprocess(text):
    txt = convert_to_unicode(text)
    # space padding punctuation
    txt = re.sub('([:@<=>;.,!?""^()])', r' \1 ', txt)
    # remove extra whitespace
    txt = re.sub(' +', ' ', txt)
    txt = txt.strip()
    # marking the start and end
    return "".join(['[start] ',txt, ' [end]'])

class simple_preprocessor(object):
    def __init__(self, tokenizer = None,  uncased = True, output_size = None):
        if not tokenizer:
            self.tokenizer = tf.keras.preprocessing.text.Tokenizer(
                        lower= uncased,
                        filters='_`{|}~\t\n'
                        )
        self.is_init = False
        self.vocab_size = 0
        self.output_size = output_size

    def fit(self, text):
        """
        input is a tensor of string or bytes
        """
        txt = [text_preprocess(sen) for sen in text]
        self.tokenizer.fit_on_texts(txt)
        self.vocab_size = len(self.tokenizer.word_index)+1
        self.is_init = True

    def __call__(self, arrays):
        """
        input has to be an array of strings
        """
        strings = [text_preprocess(text) for text in arrays]
        x = self.tokenizer.texts_to_sequences(strings)
        x = tf.keras.preprocessing.sequence.pad_sequences(x,
            maxlen = self.output_size,
            padding='post', truncating = 'post')
        mask = self.padding_mask(x)
        return x, mask

    def padding_mask(self,x):
        x = tf.cast(tf.math.equal(x, 0), tf.float32)
        return x

    def looking_ahead_mask(self):
        if 'lam' not in self.__dict__.keys():
            if size == None: 
                raise ValueError("Please define the output size before calling looking_ahead_mask!")
            mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
            self.lam =  mask  # (seq_len, seq_len)
        return self.lam

class config(pset):
    """
        the encoder cfg is a dict with option:
        L: number of layeres
        H: model depth
        A: number of attension heads
        dff: depth of hidden FF net, default will be 1024
        dropout_rate: drop rate, 0.1 as default
        positional_encoding: object for encoding the position information.
                         The default sinusoid_position_encoder will be used
                         if this part is missing.
    """

    def __init__(self,
                 input_size,
                 L= 12, H = 768, A= 12,
                 dff = 1024,
                 dropout_rate = 0.1,
                 preprocessor = None,
                 maximum_position_encoding = 10000
                 ):
        super(config, self).__init__()

        self.encoder_cfg = pset(num_layers = L, d_model = H, num_heads = A, 
                                dff = dff,
                                dropout_rate = dropout_rate,
                                positional_encoding = xtr.sinusoid_position_encoder_generator(
                                maximum_position_encoding),
                                maximum_position_encoding = maximum_position_encoding,
                                input_vocab_size = None #get from preprocessor
                               )


class model(tf.keras.Model):
    """
    preprocessor: The preprocessor used in the model, tokenizing the string input into
                      sequences.
            input_vocab_size: the input vocabulary size get from, needed for embedding layer
    """
    def __init__(self, bert_cfg, preprocessor, training = True, **kwargs):
        super(model, self).__init__(**kwargs)
        self.cfg = bert_cfg
        self.preprocessor = preprocessor
        self.encoder_cfg = bert_cfg.encoder_cfg
        if not self.preprocessor.is_init: 
            raise ValueError('ERROR: BERT model: Preprocessor has NOT been initiated yet!')
        self.encoder_cfg.update({'input_vocab_size':self.preprocessor.vocab_size})
        if not self.encoder_cfg['input_vocab_size']:
            raise ValueError('ERROR: BERT model: input_vocab_size can not be None!')

        self.encoder = xtr.Encoder(**(self.encoder_cfg))

    def build_model(self):
        """ return a keras model object"""
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text_input')
        x,m = self.preprocessor(text_input)
        output = self.encoder(x, training, m)
        return tf.Model(text_input, output)

