import numpy as np


# compute learning rate according to the CLR rule
def clr(epoch, epoch_size, step_size, lr_min, lr_max):
	''' input :	epoch --- current epoch number (zero-based)
				epoch_size --- iterations (or batches) in each epoch
				step_size --- iterations in half CLR cycle
				lr_min, lr_max --- minimum and maximum of lr
		output :	lr '''
	c = np.floor(1 + epoch*epoch_size/2/step_size)
	x = np.abs(epoch*epoch_size/step_size - 2*c + 1)
	lr = lr_min + (lr_max - lr_min)*np.max([0, 1-x])
	# over
	return lr