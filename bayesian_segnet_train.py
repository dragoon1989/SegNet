import os
import sys
import getopt

import tensorflow as tf
import numpy as np

from camvid_input import BuildPipeline
from bayesian_segnet_basic import BayesianSegNetBasic
from bayesian_segnet_basic import loss_func

from camvid_input import IMAGE_X
from camvid_input import IMAGE_Y
from camvid_input import NUM_CLASSES

import CLR

# set visible CUDA devices
os.environ['CUDA_VISIBLE_DEVICES'] = '0'		# only use one GPU is enough

# constants
train_record_path = 'CamVid/train.txt'
test_record_path = 'CamVid/test.txt'
val_record_path = 'CamVid/val.txt'

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

test_samples = 20


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

# build the BayesianSegNet basic model
with tf.name_scope('bayesian_segnet_basic_model'):
	model = BayesianSegNetBasic(input_images, NUM_CLASSES)

# train and test the model
with tf.name_scope('train_and_test'):
	# compute loss function
	loss = loss_func(input_labels, model.logits_before_softmax)
	# summary the loss
	tf.summary.scalar(name='loss', tensor=loss)

	# pixel-wise probability prediction given by the model (output of softmax layer)
	# shape = (batch_size, H, W, num_classes)
	prob = tf.nn.softmax(model.logits_before_softmax)
	
	# for training process
	# pixel-wise category predictions, dtype=tf.int32
	batch_predict = tf.argmax(prob, axis=-1, output_type=tf.int32)
	# prediction accuracy (average in one batch), dtype=tf.float32
	batch_acc = tf.reduce_mean(tf.to_float(tf.equal(tf.to_int32(input_labels), batch_predict)))
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
def train(cur_lr, sess, summary_writer, summary_op):
	'''
	input:
		cur_lr : learning rate for current epoch (scalar)
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
								   lr : cur_lr})
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
def test(sess, summary_writer):
	'''
	input :
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
	# test in batches
	while True:
		try:
			# read batch of data from testing set
			labels_val, images_val = sess.run([labels, images], feed_dict={handle:test_handle_val})
			cur_batch_size = labels_val.shape[0]
			# allocate temp buffer for batch probability prediction
			cur_prob_pred = np.zeros(shape=(cur_batch_size, IMAGE_Y, IMAGE_X, NUM_CLASSES),dtype=np.float32)
			# do the MC sampling
			for i in range(test_samples):
				# test on single batch
				prob_val, global_step_val = \
							sess.run([prob, global_step],
									 feed_dict={input_labels : labels_val,
												input_images : images_val})
				# update the probability prediction
				cur_prob_pred += prob_val/test_samples
			
			# generate the batch class prediction among all samples
			cur_class_pred = np.argmax(cur_prob_pred, axis=-1).astype(np.uint8)
			# compute all correct predictions
			labels_val = labels_val.flatten().astype(np.uint8)
			cur_class_pred = cur_class_pred.flatten()
			correctness += np.sum(a=(cur_class_pred==labels_val), dtype=np.float32)
		except tf.errors.OutOfRangeError:
			break
	# compute accuracy and loss after a whole epoch
	current_acc = correctness/test_dataset_size/IMAGE_X/IMAGE_Y
	# print and summary
	msg = 'test accuracy = %.2f%%' % (current_acc*100)
	test_acc_summary = tf.Summary(value=[tf.Summary.Value(tag='test_accuracy',simple_value=current_acc)])
	# write summary
	summary_writer.add_summary(summary=test_acc_summary, global_step=global_step_val)
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


# search config for CLR
search_for_CLR = True
# search range
lr_min = 1e-3
lr_max = 1e-2
# CLR config
epoch_size = np.ceil(train_dataset_size/train_batch_size)
step_size = 4 * epoch_size

###################### main entrance ######################
if __name__ == "__main__":
	# set tensorboard summary path
	# and load ckpt data if necessary
	ckpt_path = ''
	try:
		options, args = getopt.getopt(sys.argv[1:], '', ['logdir=', 'ckpt='])
	except getopt.GetoptError:
		print('invalid arguments!')
		sys.exit(-1)
	for option, value in options:
		if option == '--logdir':
			summary_name = value
		elif option == '--ckpt':
			ckpt_path = value
	
	if not search_for_CLR:
		# train and test the model
		best_acc = 0
		with tf.Session() as sess:
			# build tf saver
			saver = tf.train.Saver()
			# load model if necessary
			if len(ckpt_path)>0:
				saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))
			else:
				# initialize variables
				sess.run(tf.global_variables_initializer())

			# initialize IO
			sess.run(tf.local_variables_initializer())
			# build the tensorboard summary
			summary_writer = tf.summary.FileWriter(summary_path+summary_name)
			train_summary_op = tf.summary.merge_all()

			# train in epochs
			for cur_epoch in range(1, num_epochs+1):
				# compute current lr
				cur_lr = CLR.clr(epoch=cur_epoch, epoch_size=epoch_size, step_size=step_size,
								 lr_min=lr_min, lr_max=lr_max)
				# print epoch title
				print('Current epoch No.%d, learning rate = %.2e' % (cur_epoch, cur_lr))
				# train
				train(cur_lr, sess, summary_writer, train_summary_op)
				# validate
				cur_acc = test(sess, summary_writer)

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
	else:
		# find the lr_min and lr_max for CLR strategy
		# CLR config
		step_size = num_epochs * epoch_size
		# start
		with tf.Session() as sess:
			# load model if necessary
			if len(ckpt_path)>0:
				saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))
			else:
				# initialize variables
				sess.run(tf.global_variables_initializer())

			# initialize IO
			sess.run(tf.local_variables_initializer())
			# build the tensorboard summary
			summary_writer = tf.summary.FileWriter(summary_path+summary_name)
			train_summary_op = tf.summary.merge_all()

			# train in epochs
			for cur_epoch in range(1, num_epochs+1):
				# compute current lr
				cur_lr = CLR.clr(epoch = cur_epoch,
								 epoch_size = epoch_size, step_size = step_size, 
								 lr_min = lr_min,
								 lr_max = lr_max)
				# print epoch title
				print('Current epoch No.%d, learning rate = %.2e' % (cur_epoch, cur_lr))
				# train
				train(cur_lr, sess, summary_writer, train_summary_op)
				# validate
				cur_acc = test(sess, summary_writer)

		# finished
		print('++++++++++++++++++++++++++++++++++++++++')
