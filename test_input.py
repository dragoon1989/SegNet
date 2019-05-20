import tensorflow as tf
import numpy as np

from camvid_input import BuildPipeline


record_path = 'data/train.txt'
batch_size = 4

dataset = BuildPipeline(record_path=record_path,
						batch_size=batch_size,
						num_parallel_calls=4,
						num_epoch=1)

iterator = dataset.make_initializable_iterator()
dataset_handle = iterator.string_handle()
handle = tf.placeholder(tf.string, shape=[])

labels, images = iterator.get_next()

# generate image summary
tf.summary.image(name='images', tensor=images, max_outputs=batch_size)
tf.summary.image(name='labels', tensor=labels, max_outputs=batch_size)

summary_ops = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter('summary')

if __name__ == '__main__':
	with tf.Session() as sess:
		# initialize variables
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())
		# initialize iterator
		sess.run(iterator.initializer)
		handle_val = sess.run(dataset_handle)
		# consume the dataset
		summary_buff = sess.run(summary_ops, feed_dict={handle:handle_val})
		summary_writer.add_summary(summary_buff)
		#over
		print('+++++++++over+++++++++++')
