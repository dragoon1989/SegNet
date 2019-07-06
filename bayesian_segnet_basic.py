import tensorflow as tf
from tensorflow import keras
import numpy as np

import pooling



class SegNetBasic(object):

	def __init__(self, input_img, num_classes):
		self.X = input_img
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
		features = keras.layers.Activation('relu')(bn)
		# 1st pool
		features, mask1 = pooling.max_pool_with_argmax(net=features, ksize=[1,2,2,1], strides=[1,2,2,1])

		# 2nd conv
		conv2 = keras.layers.Conv2D(filters=64,
									kernel_size=(7,7), strides=(1,1), padding='SAME',
									data_format='channels_last', activation=None)(features)
		# use bn
		bn = keras.layers.BatchNormalization(axis=3)(conv2)
		# activation
		features = keras.layers.Activation('relu')(bn)
		# 2nd pool
		features, mask2 = pooling.max_pool_with_argmax(net=features, ksize=[1,2,2,1], strides=[1,2,2,1])

		# 3rd conv
		conv3 = keras.layers.Conv2D(filters=64,
									kernel_size=(7,7), strides=(1,1), padding='SAME',
									data_format='channels_last', activation=None)(features)
		# use dropout (p=0.5)
		conv3 = tf.nn.dropout(x=conv3, rate=0.5)
		# use bn
		bn = keras.layers.BatchNormalization(axis=3)(conv3)
		# activation
		features = keras.layers.Activation('relu')(bn)
		# 3rd pool
		features, mask3 = pooling.max_pool_with_argmax(net=features, ksize=[1,2,2,1], strides=[1,2,2,1])

		# 4th conv
		conv4 = keras.layers.Conv2D(filters=64,
									kernel_size=(7,7), strides=(1,1), padding='SAME',
									data_format='channels_last', activation=None)(features)
		# use dropout (p=0.5)
		conv4 = tf.nn.dropout(x=conv4, rate=0.5)
		# use bn
		bn = keras.layers.BatchNormalization(axis=3)(conv4)
		# activation
		features = keras.layers.Activation('relu')(bn)
		# 4th pool
		#features, mask4 = pooling.max_pool_with_argmax(net=features, ksize=[1,2,2,1], strides=[1,2,2,1])

		# 1st upsample
		#features = pooling.unpool(features, mask4, ksize=[1, 2, 2, 1])
		# 1st 'deconv'
		deconv1 = keras.layers.Conv2D(filters=64,
									kernel_size=(7,7), strides=(1,1), padding='SAME',
									data_format='channels_last', activation=None)(features)
		# use dropout (p=0.5)
		deconv1 = tf.nn.dropout(x=deconv1, rate=0.5)
		
		# 2nd upsample
		features = pooling.unpool(features, mask3, ksize=[1, 2, 2, 1])
		# 2nd 'deconv'
		deconv2 = keras.layers.Conv2D(filters=64,
									kernel_size=(7,7), strides=(1,1), padding='SAME',
									data_format='channels_last', activation=None)(features)
		# use dropout (p=0.5)
		deconv2 = tf.nn.dropout(x=deconv2, rate=0.5)
		
		# 3rd upsample
		features = pooling.unpool(features, mask2, ksize=[1, 2, 2, 1])
		# 3rd 'deconv'
		deconv3 = keras.layers.Conv2D(filters=64,
									kernel_size=(7,7), strides=(1,1), padding='SAME',
									data_format='channels_last', activation=None)(features)

		# 4th upsample
		features = pooling.unpool(features, mask1, ksize=[1, 2, 2, 1])
		# 4th 'deconv', output classes predictions
		deconv4 = keras.layers.Conv2D(filters=self.NUM_CLASSES,
									kernel_size=(7,7), strides=(1,1), padding='SAME',
									data_format='channels_last', activation=None)(features)
		# no softmax
		self.logits_before_softmax = deconv4

# compute loss function (cross entropy in all pixels)
def loss_func(labels, logits_before_softmax):
	''' input:	labels --- sparse labels with shape [batch, H, W], dtype = tf.uint8
				logits_before_softmax --- logits before softmax with shape [batch, H, W, NUM_CLASSES], dtype = tf.float
		output:	loss --- scalar cross entropy, dtype = tf.float '''
	batch_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.to_int32(labels),
												  logits=logits_before_softmax,
												  name='loss')
	# reduce the batch loss to a mean scalar
	loss = tf.reduce_mean(batch_loss)
	# over
	return loss

# compute loss function with rebalancing weights
def weighted_loss_func(labels, logits_before_softmax, weights):
	''' input:	labels --- sparse labels with shape [batch, H, W], dtype = tf.uint8
				logits_before_softmax --- logits before softmax with shape [batch, H, W, NUM_CLASSES], dtype = tf.float
				weights --- 1D array, dtype = tf.float32
		output:	loss --- scalar weighted cross entropy, dtype = tf.float32 '''
	# get the NUM_CLASSES
	num_classes = tf.shape(weights)[0]
	# flatten the input labels to 1D vector with length =batch*H*W
	B = tf.shape(labels)[0]
	H = tf.shape(labels)[1]
	W = tf.shape(labels)[2]
	labels = tf.reshape(labels, shape=[B*H*W])
	# convert input sparse labels to one-hot codes (shape = [B*H*W, NUM_CLASSES])
	labels = tf.one_hot(indices=labels, depth=num_classes, dtype=tf.float32)
	# compute logits after softmax
	logits = tf.nn.softmax(logits_before_softmax)
	# reshape the logits to [B*H*W, NUM_CLASSES]
	logits = tf.reshape(logits, shape=[B*H*W, num_classes])
	# compute weighted cross entropy
	batch_loss = tf.reduce_sum(-tf.multiply(labels*tf.log(logits + 1e-9), weights),axis=[1])
	# reduce the batch loss to loss
	loss = tf.reduce_mean(batch_loss)
	# over
	return loss
