import os
import sys
import getopt

import tensorflow as tf
import numpy as np

from camvid_input import BuildPipeline
from segnet_basic import SegNetBasic
from segnet_basic import loss_func
from segnet_basic import weighted_loss_func

from camvid_input import IMAGE_X
from camvid_input import IMAGE_Y
from camvid_input import NUM_CLASSES

# set visible CUDA devices
os.environ['CUDA_VISIBLE_DEVICES'] = '0'		# only use one GPU is enough

# constants
train_record_path = 'CamVid/train.txt'
test_record_path = 'CamVid/test.txt'
val_record_path = 'CamVid/val.txt'

train_weights_path = 'CamVid/train_weights.txt'
test_weights_path = 'CamVid/test_weights.txt'

summary_path = './tensorboard/'
summary_name = 'summary-default'    # tensorboard default summary dir

model_path = './ckpts/'
best_model_ckpt = 'best.ckpt'		# check point path

train_dataset_size = 367
test_dataset_size = 233
val_dataset_size = 101

# hyperparameters
train_batch_size = 16
test_batch_size = 16
val_batch_size = 20

num_epochs = 50
lr0 = 1e-5


# build input pipeline
with tf.name_scope('input'):
	# train data
	train_dataset = BuildPipeline(record_path=train_record_path,
						batch_size=train_batch_size,
						num_parallel_calls=4,
						num_epoch=1)
	train_iterator = train_dataset.make_initializable_iterator()
	# test data
	test_dataset = BuildPipeline(record_path=test_record_path,
						batch_size=test_batch_size,
						num_parallel_calls=4,
						num_epoch=1)
	test_iterator = test_dataset.make_initializable_iterator()
	# handle of pipelines
	train_handle = train_iterator.string_handle()
	test_handle = test_iterator.string_handle()
	# build public data entrance
	handle = tf.placeholder(tf.string, shape=[])
	iterator = tf.data.Iterator.from_string_handle(handle, train_iterator.output_types)
	labels, images = iterator.get_next()
	# build placeholder for model input and output
	# batch of data will be fed to these placeholders
	input_images = tf.placeholder(tf.float32, shape=(None, IMAGE_Y, IMAGE_X, 3))
	input_labels = tf.placeholder(tf.uint8, shape=(None, IMAGE_Y, IMAGE_X))

# set global step counter
global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')

# build the SegNet basic model
with tf.name_scope('segnet_basic_model'):
	model = SegNetBasic(input_images, NUM_CLASSES)

# train and test the model
with tf.name_scope('train_and_test'):
	# placeholder for loss weights
	loss_weights = tf.placeholder(tf.float32, shape=[NUM_CLASSES])
	# compute loss function
	#loss = loss_func(input_labels, model.logits_before_softmax)
	# compute weighted loss function
	loss = weighted_loss_func(input_labels, model.logits_before_softmax, loss_weights)
	# summary the loss
	tf.summary.scalar(name='loss', tensor=loss)

	# evaluate batch accuracy
	# batch predictions, dtype=tf.float32
	batch_predict = tf.to_float(tf.argmax(tf.nn.softmax(model.logits_before_softmax), axis=-1))
	# accuracy
	batch_acc = tf.reduce_mean(tf.to_float(tf.equal(tf.to_float(input_labels), batch_predict)))
	# summary the batch accuracy
	tf.summary.scalar(name='batch_acc', tensor=batch_acc)

	# optimize model parameters
	with tf.name_scope('optimization'):
		# placeholder to control learning rate
		lr = tf.placeholder(dtype=tf.float32, shape=[], name='learning_rate')
		# use Adam
		train_op = tf.train.AdamOptimizer(learning_rate=lr,
									   beta1=0.9,
									   beta2=0.999,
									   epsilon=1e-08).minimize(loss, global_step=global_step)

# build the training process
def train(cur_lr, train_loss_weights, sess, summary_writer, summary_op):
	'''
	input:
		cur_lr : learning rate for current epoch (scalar)
		train_loss_weights : loss weights for training set
		sess : tf session to run the training process
		summary_writer : summary writer
		summary_op : summary to write in training process
	'''
	# get iterator handles
	train_handle_val = sess.run(train_handle)
	# initialize iterator
	sess.run(train_iterator.initializer)
	# training loop
	current_batch = 0
	while True:
		try:
			# read batch of data from training set
			labels_val, images_val = sess.run([labels, images], feed_dict={handle:train_handle_val})
			# feed this batch to AlexNet
			_, loss_val, batch_acc_val, global_step_val, summary_buff = \
				sess.run([train_op, loss, batch_acc, global_step, summary_op],
						feed_dict={input_labels : labels_val,
								   input_images : images_val,
								   lr : cur_lr,
								   loss_weights : train_loss_weights})
			current_batch += 1
			# print indication info
			if current_batch % 4 == 0:
				msg = '\tbatch number = %d, loss = %.2f, train accuracy = %.2f%%' % \
						(current_batch, loss_val, batch_acc_val*100)
				print(msg)
				# write train summary
				summary_writer.add_summary(summary=summary_buff, global_step=global_step_val)
		except tf.errors.OutOfRangeError:
			break
	# over

# build the test process
def test(test_loss_weights, sess, summary_writer):
	'''
	input :
		test_loss_weights : loss weights for test set
		sess : tf session to run the validation
		summary_writer : summary writer
		test_summary_op : summary to be writen in test process
	'''
	# get iterator handle
	test_handle_val = sess.run(test_handle)
	# initialize iterator
	sess.run(test_iterator.initializer)
	# validation loop
	correctness = 0
	loss_val = 0
	while True:
		try:
			# read batch of data from testing set
			labels_val, images_val = sess.run([labels, images], feed_dict={handle:test_handle_val})
			cur_batch_size = labels_val.shape[0]
			# test on single batch
			batch_predict_val, batch_loss_val, global_step_val = \
						sess.run([batch_predict, loss, global_step],
								 feed_dict={input_labels : labels_val,
											input_images : images_val,
											loss_weights : test_loss_weights})
			# for labels and predictions are all N-d arrays, we must flatten them first ...
			labels_val = labels_val.flatten().astype(np.float32)
			batch_predict_val = batch_predict_val.flatten().astype(np.float32)

			correctness += np.asscalar(np.sum(a=(batch_predict_val==labels_val), dtype=np.float32))
			loss_val += np.asscalar(batch_loss_val*cur_batch_size*IMAGE_X*IMAGE_Y)
		except tf.errors.OutOfRangeError:
			break
	# compute accuracy and loss after a whole epoch
	current_acc = correctness/test_dataset_size/IMAGE_X/IMAGE_Y
	loss_val /= test_dataset_size/IMAGE_X/IMAGE_Y
	# print and summary
	msg = 'test accuracy = %.2f%%' % (current_acc*100)
	test_acc_summary = tf.Summary(value=[tf.Summary.Value(tag='test_accuracy',simple_value=current_acc)])
	test_loss_summary = tf.Summary(value=[tf.Summary.Value(tag='test_loss', simple_value=loss_val)])
	# write summary
	summary_writer.add_summary(summary=test_acc_summary, global_step=global_step_val)
	summary_writer.add_summary(summary=test_loss_summary, global_step=global_step_val)
	# print message
	print(msg)
	# over
	return current_acc

# simple function to adjust learning rate between epochs
def update_learning_rate(cur_epoch):
	'''
	input:
		epoch : current No. of epoch
	output:
		cur_lr : learning rate for current epoch
	'''
	cur_lr = lr0
	if cur_epoch > 10:
		cur_lr = lr0/10
	if cur_epoch >20:
		cur_lr = lr0/100
	if cur_epoch >30:
		cur_lr = lr0/1000
	if cur_epoch >40:
		cur_lr = lr0/2000
	# over
	return cur_lr



###################### main entrance ######################
if __name__ == "__main__":
	# set tensorboard summary path
	try:
		options, args = getopt.getopt(sys.argv[1:], '', ['logdir='])
	except getopt.GetoptError:
		print('invalid arguments!')
		sys.exit(-1)
	for option, value in options:
		if option == '--logdir':
			summary_name = value

	# read in the loss weights
	train_loss_weights = np.loadtxt(train_weights_path)
	test_loss_weights =np.loadtxt(test_weights_path)
	# train and test the model
	cur_lr = lr0
	best_acc = 0
	with tf.Session() as sess:
		# initialize variables
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())

		# initialize IO
		# build tf saver
		saver = tf.train.Saver()
		# build the tensorboard summary
		summary_writer = tf.summary.FileWriter(summary_path+summary_name)
		train_summary_op = tf.summary.merge_all()

		# train in epochs
		for cur_epoch in range(1, num_epochs+1):
			# print epoch title
			print('Current epoch No.%d, learning rate = %.2e' % (cur_epoch, cur_lr))
			# train
			train(cur_lr, train_loss_weights, sess, summary_writer, train_summary_op)
			# validate
			cur_acc = test(test_loss_weights, sess, summary_writer)
			# update learning rate if necessary
			cur_lr = update_learning_rate(cur_epoch)

			if cur_acc > best_acc:
				# save check point
				saver.save(sess=sess,save_path=model_path+best_model_ckpt)
				# print message
				print('model improved, save the ckpt.')
				# update best loss
				best_acc = cur_acc
			else:
				# print message
				print('model not improved.')
	# finished
	print('++++++++++++++++++++++++++++++++++++++++')
	print('best accuracy = %.2f%%.'%(best_acc*100))
