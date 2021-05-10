
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import module.BBox as bbox
EPSILON = 1e-8
class Anchor(object):
    def __init__(self, input_size, 
                 base_scale,
                 feature_size,
                 num_scales = 3,
                 aspect_ratios = [0.5, 1, 2],
                 ):
        """Auxiliary object anchor for encoding/decoding the anchor box and class
        Args:
            input_size: `int` pixel size of the inputs, assuming the input is a square tensor
            feature_size: `[int]` size array of the layer anchored to.
                since the input and the layer is assumed to be square, the size
                is fine to be a single integer.
            num scales: `int`, total scales = [ 2**(1.0/n) for n in range(num_scales)]
                     
            aspect_ratio: `float array`, the aspect ratios = weight/height of box
        """
        self.base_scale = base_scale
        self.window_size = [int(input_size//size) for size in feature_size]
        self.num_scales = num_scales
        self.aspect_ratios = aspect_ratios
        self.input_size = input_size
        self.nratio = len(aspect_ratios)
        self.feature_size = feature_size
        #  size:  the number of the anchored box 
        #  it equal to number of the aspect_ratios * the number scales added * strides
        self.nRes = len(self.feature_size) # number of resolution level
        self.size = self.nratio * self.num_scales
        self.boxes= self.generate_box_grid()

    def generate_box_grid(self):
        scales = [0.5**n for n in range(self.num_scales)]
        boxes_all = []
        for l in range(len(self.feature_size)):
            boxes_l = []
            for scale in scales:
                for aspect_ratio in self.aspect_ratios:
                    st = self.window_size[l]
                    base_box_size = self.window_size[l]*scale*self.base_scale
                    #print(base_box_size)
                    aspect_x = aspect_ratio**0.5
                    aspect_y = aspect_ratio**-0.5
                    half_anchor_size_x = base_box_size * aspect_x / 2.0
                    half_anchor_size_y = base_box_size * aspect_y / 2.0
                    x = tf.range(st / 2, self.input_size, st)
                    y = tf.range(st / 2, self.input_size, st)
                    xv, yv = tf.meshgrid(x, y)
                    xv = tf.cast(tf.reshape(xv, [-1]), dtype=tf.float32)
                    yv = tf.cast(tf.reshape(yv, [-1]), dtype=tf.float32)
                    # Tensor shape Nx4.
                    boxes = tf.stack([
                        xv - half_anchor_size_x, yv - half_anchor_size_y,
                        xv + half_anchor_size_x, yv + half_anchor_size_y
                        #yv - half_anchor_size_y, xv - half_anchor_size_x,
                        #yv + half_anchor_size_y, xv + half_anchor_size_x
                    ],
                                     axis=1)
                    boxes_l.append(boxes)
            # Concat anchors on the same level to tensor shape NxAx4.
            boxes_l = tf.stack(boxes_l, axis=1)
            boxes_l = tf.reshape(boxes_l, [-1, 4])
            boxes_all.append(boxes_l)
        return tf.concat(boxes_all, axis=0)

    def encode_box(self, boxes, gt_boxes):
        """
        change the pixel coordiantes [x0,y0,x1,y1] of boxes into relative coordinates wr. anchor boxes
        """
        xc_a, yc_a, wa, ha = bbox.pcoordinate_to_ccoordinate(boxes)
        xc, yc, w, h= bbox.pcoordinate_to_ccoordinate(gt_boxes)
        ha+=EPSILON
        wa+=EPSILON
        h+=EPSILON
        h+=EPSILON
        tx = (xc- xc_a) / wa
        ty = (yc- yc_a) / ha
        tw = tf.math.log(w / wa)
        th = tf.math.log(h / ha)
        # Scales location targets as used in paper for joint training.
        #if self._scale_factors:
        #    ty *= self._scale_factors[0]
        #    tx *= self._scale_factors[1]
        #    th *= self._scale_factors[2]
        #    tw *= self._scale_factors[3]
        return tf.transpose(a=tf.stack([tx, ty, tw, th]))
        

    def generate_labels(self,x, gt_boxes,
                            gt_labels,
                            match_threshold = 0.5,
                            unmatch_threshold = 0.4):
        """ 
        associate the label to anchor boxes by matching the anchor boxes to the 
        ground truth box gt_boxes from training samples,
        matching is implemented by using a arg max of the iou between the grid box and gt_boxes.
        generate classification labels [N, M], and the anchor boxes corrdinates [N,4],
        where N is # of anchors and M is # of classes.

        The default coordinaate values for the unmatched box or the ignored box is [0,0,0,0]
        """
        indices = bbox.box_iou_match(x = x, gt_boxes = gt_boxes, 
                           match_threshold = match_threshold, 
                           unmatch_threshold = unmatch_threshold)
        # the matched gt_boxes
        match = bbox.generate_target(labels = gt_boxes, indices = indices, 
                                     unmatched_value = tf.zeros(4),
                                     ignored_value = tf.zeros(4))
        reg = self.encode_box(boxes=self.boxes, gt_boxes = match)
        cls = bbox.generate_target(labels = gt_labels, indices = indices, 
                                     unmatched_value = -2,
                                     ignored_value = -1)
        return reg, cls

class AnchorRegression(tf.keras.layers.Layer):
    """
    base layer for box/classification regressor
    The only difference between the box/cls is the output channel from box = 4*cls.

    For classification: 
        size = aspect size * scales * nclass
    For box regression: 
        size = aspect size * scales * 4
    
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

        self.cfg = {'name':name, 'anchor_size':anchor_size,
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
        """
        The features should be an array of multi level features:
        features = [f1, f2, ...]
        """
        output = []
        for x in features:
            x = self.layer(x)
            output.append(self.box(x))
        return output

    def get_config(self):
        cfg = super(AnchorRegression, self).get_config()
        cfg.update(self.cfg)
        return cfg

class Head(tf.keras.layers.Layer):
    """
    nclass: # of class types need to be detected
    anchor_size : (aspect ratio # )*(scales #), should be obtained from Anchor.size
    The Convolution layer configurations are set for the FCN used in this head
    """
    def __init__(self,
                 name,
                 nclass,
                 anchor_size,
                 filters = 256,
                 activation = 'relu',
                 kernel_size = 3,
                 strides = 1,
                 padding = 'same',
                 use_batch_norm = True, **kwargs):
        super(Head, self).__init__(**kwargs)
        self.cfg = {'name':name,
                    'anchor_size':anchor_size,
                    'filters':filters,
                    'activation':activation,
                    'kernel_size':kernel_size,
                    'strides':strides,
                    'padding':padding,
                    'use_batch_norm':use_batch_norm}
    
        #For classification: 
        #    size = aspect size * scales * nclass
        self.cfg['anchor_size'] = nclass*self.cfg['anchor_size']
        self.cls_layer = AnchorRegression(**self.cfg)
        #For box regression: 
        #    size = aspect size * scales * 4
        self.cfg_box = dict(self.cfg)
        self.cfg_box['anchor_size'] = 4*self.cfg_box['anchor_size']
        self.box_layer = AnchorRegression(**self.cfg_box)

    def __call__(self, x, taining = None):
        ycls = self.cls_layer(x)
        ybox = self.box_layer(x)
        return {'label':ycls,'box':ybox}

    def get_config(self):
        cfg = super(Head, self).get_config()
        cfg.update(self.cfg)
        return cfg

