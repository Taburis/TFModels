
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

class Anchor(object):
    def __init__(self, nclass, scales, aspect_ratios = [0.5, 1, 2])
        """
        Args:
            nclass: the total number of the class inputs
            scales: `int`, the number of different box scale
            aspect_ratio: `float array`, the aspect ratios = weight/height of box
        """
        self.nclass =nclass
        self.scale = scale
        self.aspect_ratios = aspect_ratios
        self.size = len(self.aspect_ratios) * nclass


class AnchorRegression(tf.keras.layers.Layer):
    """
    base layer for box/classification regressor
    The only difference between the box/cls is the output channel from box = 4*cls.
    The anchor size is the channel number for each neural in feature map, can be obtained
    from the `Anchor.size` from the Anchor object above.
    
    Input to this layer should be an array of feature tensor in case the feature pyramid is 
    applied. So is the output matching to the input feature level
    """
    def __init__(self, 
                 name,
                 anchor_size,
                 filters = 256,
                 activation = 'relu',
                 kernel_size = 3,
                 strides = 1,
                 padding = 'same',
                 use_batch_norm = True, **kwargs):
        super(AnchorRegression, self).__init__(**kwargs)

        self.cfg = {'name':name,
                    'anchor_size':anchor_size,
                    'filters':filters,
                    'activation':activation,
                    'kernel_size':kernel_size,
                    'strides':strides,
                    'padding':padding,
                    'use_batch_norm':use_batch_norm}

        self.layer = tf.keras.layers.Conv2D(filters, 
                                            kernel_size = kernel_size,
                                            strides = strides, 
                                            padding = padding,
                                            activation = activation, 
                                            bias_initializer=tf.zeros_initializer(),
                                            use_bias = False)
        if use_batch_norm: 
            self.norm = tf.keras.layers.BatchNormalization()

        self.box = tf.keras.layers.Conv2D(filters = anchor_size,
                                          kernel_size = kernel_size,
                                          strides = strides, 
                                          padding = padding,
                                          activation = activation, 
                                          bias_initializer=tf.zeros_initializer(),
                                          use_bias = False)

    def __call__(self, features, traing = None):
        output = []
        for x in features:
            x = self.layer(x)
            output.append(self.box(x))
        return output

    def get_config(self):
        cfg = super(AnchorRegression, self).get_config()
        cfg.update(self.cfg)
        return cfg

class RPNHead(tf.keras.layers.Layer):
    def __init_(self,
                name,
                anchor_size,
                filters = 256,
                activation = 'relu',
                kernel_size = 3,
                strides = 1,
                padding = 'same',
                use_batch_norm = True, **kwargs):
        super(RPNHead, self).__init__(**kwargs)
        self.cfg = {'name':name,
                    'anchor_size':anchor_size,
                    'filters':filters,
                    'activation':activation,
                    'kernel_size':kernel_size,
                    'strides':strides,
                    'padding':padding,
                    'use_batch_norm':use_batch_norm}
         
        self.cfg_box = dict(self.cfg)
        self.cfg_box['anchor_size'] = 4*self.cfg_box['anchor_size']
        self.box_layer = AnchorRegression(**self.cfg_box)
        self.cfg_layer = AnchorRegression(**self.cfg)

    def __call__(self, x, taining = None):
        ycls = self.cls_layer(x)
        ybox = self.box_layer(x)
        return {'label':ycls,'box':ybox}

    def get_config(self):
        cfg = super(RPNHead, self).get_config()
        cfg.update(self.cfg)
        return cfg


