
import layers as xlayers
import tensorflow as tf

def sequential_block(nlayer, module, cfg, trainable = True):
    """ stacks sub-blocks to form a new sequential blocks
    """
    module_fn = module(cfg, trainable)	
    def sequential_block_imp(inputs):
        x = inputs
        for _ in range(nlayer):
            x = module_fn(x)
        return x
    return sequential_block_imp

def sequential_conv2d(cfg, trainable = True):
    """stacked 2D convolution layers
    cfg {
        conv2d : 'dict', conv2d configuration;
        trainable : 'bool', if this block is trainable;
        nlayer : 'int', number of conv2d for stacking;
        use_residual: 'bool', if the residual learning will be added,
                    the shortcut will connect the input and the output
    }
    """
    cfg_conv2d = cfg.conv2d
    cfg_conv2d['trainable'] = trainable
    nlayer= cfg.nlayer
    use_residual = cfg.residual_learning
    def sequential_conv2d_imp(inputs):
        x = inputs
        shortcut = inputs
        for i in range( nlayer):
            x = tf.keras.layers.Conv2D(**cfg_conv2d)(x)
        if use_residual : 
            x = xlayers.residual2D(trainable = trainable)(shortcut, x)
        return x
    return sequential_conv2d_imp

