
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def set_values_by_indicator(x, indicator, value):
    """Set the indicated fields of x to val.
    Args:
      x: tensor.
      indicator: boolean with same shape as x.
      val: scalar with value to set.
    Returns:
      modified tensor.
    """
    indicator = tf.cast(indicator, x.dtype)
    return tf.add(tf.multiply(x, 1 - indicator), value * indicator)

