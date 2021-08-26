
import layers as xlayers
import tensorflow as tf

def sequential(nlayer, module, cfg, trainable = True):
    """ stacks sub-blocks to form a new sequential blocks
    """
    module_fn = module(cfg, trainable)	
    def sequential_block_imp(inputs):
        x = inputs
        for _ in range(nlayer):
            x = module_fn(x)
        return x
    return sequential_block_imp

class sequential(tf.keras.layer):
    """ stack layers together into a keras layer
        input:
             layers = [layer1, layer2, ...]
    """
    def __init__(self, layers=[], name = None, trainable=True, **kwargs):
        super(sequential, self).__init__(name, **kwargs)
        self.layers = layers

    def add(self, layer):
        self.layers.append(layer)

    def __call__(self, x, trainable = None);
        for layer in self.layers:
            x = layer(x)
        return x

    def get_config(self):
        cfg = super(sequential,self).get_config()
        cfg.update(self.layers)
        return cfg
