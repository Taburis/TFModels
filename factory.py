
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class factory(object):
    def __init__(self, model):
        self.model = model
        self.outputs = model.outputs
    
    def loss(self, y_true, y_pred):
        return self.loss_fn(y_true= y_true, y_pred = y_pred)
    
    def training_iter_step(self, inputs):
        x, labels = inputs
        
        with tf.GradientTape() as tape:
            y_ = self.model(x)
            loss = self.loss_fn(y_true= labels, y_pred = y_)
        
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss
    
    def training_step(self, iterator, num_steps):
        if not isinstance(num_steps, tf.Tensor):
            num_steps = tf.convert_to_tensor(num_steps, dtype = tf.int32)
        for _ in tf.range(num_steps -1):
            print(self.training_iter_step(next(iterator)))
        return
