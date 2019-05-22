import tensorflow as tf
import numpy as np

# constants
IMAGE_X = 480
IMAGE_Y = 360
NUM_CLASSES = 12

# do lcn on an input image
def __lcn(image):
	''' input: image	image in format HWC, with dtype=float
		output: filtered_img	image in format HWC, with dtype=float'''
	_mean, _var = tf.nn.moments(x=image, axes=[0,1])
	filtered_img = (image - _mean)/tf.sqrt(_var)
	return filtered_img

# read single .png image
def __read_png_image(img_path):
	''' input: img_path	path to the png image file
		output: image	the image with format HWC, dtype=uint8'''
	# the decoded image is in format HWC
	image = tf.image.decode_png(tf.read_file(img_path), dtype=tf.uint8)
	return image

# read single example from files
def __read_single_example(lab_path, img_path):
	''' input: lab_path	path to the labeled png image file
			   img_path	path to the png image file
		output: label	labeled image with shape [H,W], dtype=tf.uint8
				image	image with shape [H,W,3], dtype=tf.float'''
	# the label is in format HW (dtype=uint8)
	label = __read_png_image(lab_path)
	# the image is in format HWC (dtype=uint8)
	image = __read_png_image(img_path)
	# reshape and cast type
	label = tf.reshape(label, shape=(IMAGE_Y, IMAGE_X))
	image = tf.to_float(image)
	return label, image

# parse single record from a text file
def __parse_single_record(record):
	''' input: record	string tensor (scalar)
		output: label --- tensor with shape [H,W], dtype=tf.uint8
				image --- tensor with shape [H,W,3], dtype=tf.float '''
	segments = tf.string_split(source=[record], delimiter=' ')
	lab_path = segments.values[1]
	img_path = segments.values[0]
	label, image = __read_single_example(lab_path, img_path)
	# do the LCN
	image = __lcn(image)
	# over
	return label, image

# build a pipeline
def BuildPipeline(record_path,
				  batch_size,
				  num_parallel_calls=1,
				  num_epoch=1):
	''' input: record_path	path to the record file
			   batch_size
			   num_parallel_calls
			   num_epoch
		output: dataset		a dataset '''
	# open the record file
	dataset = tf.data.TextLineDataset(record_path)
	# parse each record
	dataset = dataset.map(map_func=__parse_single_record, num_parallel_calls=num_parallel_calls)
	# set the epoch
	dataset = dataset.repeat(num_epoch)
	# shuffle
	dataset = dataset.shuffle(buffer_size=10*batch_size)
	# batch
	dataset = dataset.batch(batch_size)
	# prefetch 1 batch
	dataset = dataset.prefetch(buffer_size=1)
	# over
	return dataset
