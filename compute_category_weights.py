import numpy as np
import cv2


# read an image label file and compute category pixels
def __analyze_single_label__(file_path, categories):
	'''
	input: file_path	input label file path (python string scalar)
		   categories	list of legal categories (1-D python list / numpy array)
	output:num_pix		num of pixels for each category (1-D numpy array, dtype=np.int64)
		   img_size		size of label image file
	'''
	# get num of categories
	n = len(categories)
	# read lable image (HW format)
	label_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
	# initialize num_pix as all zeros
	num_pix = np.zeros(n)
	# traverse the input label as a flat array
	label_img = label_img.ravel()
	img_size = len(label_img)
	for p in label_img:
		num_pix[p] += 1
	# over
	return num_pix, img_size

# read a record file and get label file paths
def __analyze_record__(record_path):
	'''
	input: record_path	input record file path
	output:label_paths	label file paths (1-D python string list)
	'''
	with open(record_path, 'r') as f:
		text = f.read()
	# split the text
	paths = text.split()
	# only the 2nd column is needed
	n = len(paths)
	label_paths = [paths[i] for i in range(1,n,2)]
	# over
	return label_paths

# compute categorical data
def ComputCategoricalData(record_path, categories):
	'''
	input: record_path	input record file path
	       categories	list of legal categories (1-D python list / numpy array)
	output:freq	1-D numpy array of categorical frequencies (dtype=np.float32)
		   med	median of freq
	'''
	label_paths = __analyze_record__(record_path)
	n = len(categories)
	pixels = np.zeros(n)
	sizes = np.zeros(n)
	for path in label_paths:
		num_pix, img_size = __analyze_single_label__(path, categories)
		pixels += num_pix
		mask = num_pix > 0
		sizes += mask.astype(np.int32)*img_size
	# compute categorical frequencies
	freq = pixels.astype(np.float32)/sizes
	# compute frequencies median
	med = np.median(freq)
	# over
	return freq, med
