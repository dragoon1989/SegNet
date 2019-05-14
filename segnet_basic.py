import tensorflow as tf
from tensorflow import keras
import numpy as np



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
		features = keras.
		
		# 2nd conv
		conv2 = keras.layers.Conv2D(filters=64,
									kernel_size=(7,7), strides=(1,1), padding='SAME',
									data_format='channels_last', activation=None)
		