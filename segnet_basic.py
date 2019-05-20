import tensorflow as tf
from tensorflow import keras
import numpy as np

import tf_unpool



class SegnetBasic(object):
	
	def __init__(self, input_img, num_classes):
		self.X = images
		self.NUM_CLASSES = num_classes
		self.create()
	
	def create(self):
		# 1st conv
		conv1 = keras.layers.Conv2D(filters=64, 
									kernel_size=(7,7), strides=(1,1), padding='SAME',
									data_format='channels_last', activation=None)(self.X)
		# use bn
		bn = keras.layers.BatchNormalization(axis=3)(conv1)
		# activation
		features = keras.Activation('relu')(bn)
		# 1st pool
		features, mask1 = tf_unpool.max_pool_with_argmax(features, 2)
		
		# 2nd conv
		conv2 = keras.layers.Conv2D(filters=64,
									kernel_size=(7,7), strides=(1,1), padding='SAME',
									data_format='channels_last', activation=None)(features)
		# use bn
		bn = keras.layers.BatchNormalization(axis=3)(conv2)
		# activation
		features = keras.Activation('relu')(bn)
		# 2nd pool
		features, mask2 = tf_unpool.max_pool_with_argmax(features, 2)
		
		# 3rd conv
		conv3 = keras.layers.Conv2D(filters=64,
									kernel_size=(7,7), strides=(1,1), padding='SAME',
									data_format='channels_last', activation=None)(features)
		# use bn
		bn = keras.layers.BatchNormalization(axis=3)(conv3)
		# activation
		features = keras.Activation('relu')(bn)
		# 3rd pool
		features, mask3 = tf_unpool.max_pool_with_argmax(features, 2)
		
		# 4th conv
		conv4 = keras.layers.Conv2D(filters=64,
									kernel_size=(7,7), strides=(1,1), padding='SAME',
									data_format='channels_last', activation=None)(features)
		# use bn
		bn = keras.layers.BatchNormalization(axis=3)(conv4)
		# activation
		features = keras.Activation('relu')(bn)
		# 3rd pool
		features, mask4 = tf_unpool.max_pool_with_argmax(features, 2)
		
		# 1st upsample
		features = tf_unpool.un_max_pool(features, mask4, 2)
		# 1st 'deconv'
		deconv1 = keras.layers.Conv2D(filters=64,
									kernel_size=(7,7), strides=(1,1), padding='SAME',
									data_format='channels_last', activation=None)(features)
		# 2nd upsample
		features = tf_unpool.un_max_pool(deconv1, mask3, 2)
		# 2nd 'deconv'
		deconv2 = keras.layers.Conv2D(filters=64,
									kernel_size=(7,7), strides=(1,1), padding='SAME',
									data_format='channels_last', activation=None)(features)
		
		# 3rd upsample
		features = tf_unpool.un_max_pool(deconv2, mask2, 2)
		# 3rd 'deconv'
		deconv3 = keras.layers.Conv2D(filters=64,
									kernel_size=(7,7), strides=(1,1), padding='SAME',
									data_format='channels_last', activation=None)(features)
		
		# 4th upsample
		features = tf_unpool.un_max_pool(deconv3, mask1, 2)
		# 4th 'deconv', output classes predictions
		deconv4 = keras.layers.Conv2D(filters=self.NUM_CLASSES,
									kernel_size=(7,7), strides=(1,1), padding='SAME',
									data_format='channels_last', activation=None)(features)
		# no softmax
		self.logits_before_softmax = deconv4
		# softmax along channels
		#self.logits = keras.layers.Softmax(axis=-1)(deconv4)
	
# compute loss function (cross entropy in all pixels)
def loss_func(labels, logits_before_softmax):
	''' input:	labels --- sparse labels with shape [batch, H, W], dtype = tf.uint8
				logits_before_softmax --- logits before softmax with shape [batch, H, W, NUM_CLASSES], dtype = tf.float
		output:	loss --- cross entropy with shape [batch, H, W], dtype = tf.float '''
	batch_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, 
												  logits=logits_before_softmax, 
												  name='loss')
	# reduce the batch loss to a mean scalar
	loss = tf.reduce_mean(batch_loss)