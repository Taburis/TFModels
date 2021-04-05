
import tensorflow as tf

class residual2D_layer(tf.keras.layers.Layer):
	""" Known as shortcut in residual learning. 
	"""
	def __init__(self, 
				 use_projection = False,
				 trainable = True, **kwargs):
		super(residual2D_layer, self).__init__(self, **kwargs)
		self.use_projection = use_projection
		self.trainable = trainable

	def call(self, x, r, dim = 3):
		"""
		Args: 
		x: the shortcut from inputs
		r: residual inputs
		dim: which dimension is the filters
		"""
		shortcut = x
		nfilter = r.shape[dim]
		if not r.shape[dim] == x.shape[dim] : 
			self.use_projection = True
		if self.use_projection : 
			self.layer = tf.keras.layers.Conv2D(filters = nfilter, kernel_size = 1, strides = 1, padding = 'valid', trainable = self.trainable)
			shortcut = self.layer(x)
#			print('filters for shortcut: ', nfilter, shortcut.shape[dim])
		return shortcut + r

	def get_config(self):
		return {'use_projection':self.use_projection, 'trainable':self.trainable}


