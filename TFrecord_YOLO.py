import numpy as np
from PIL import Image
import tensorflow as tf
import io
import hashlib

def _byte_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _byte_list_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_list_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_list_feature(value):
	return tf.train.Feature(float_list=tf.train.FloatList(value=value))

num = 0

#change the category here
all_categ = ['bird','cat','cow','dog']

tfrecords_filename = 'tf_train.record'

writer = tf.python_io.TFRecordWriter(tfrecords_filename)

direc_file = '' # files that list all image locations
direc = []
with open (direc_file) as f:
	for line in f:
		direc.append(line[:-1])

for line in direc:
	if num % 1000 == 0:
		print(num+ '...')
	num = num + 1
	img_path = line
	filename = line[18:-4]
	annotation_path = line[:-4] + '.txt'

	with tf.gfile.GFile(img_path,'rb') as fid:
		encoded_jpg = fid.read()
	encoded_jpg_io = io.BytesIO(encoded_jpg)

	key = hashlib.sha256(encoded_jpg).hexdigest()

	with open (annotation_path,'r') as txtf:
		annotation_string = txtf.read()
	annotation_plt = annotation_string.split()

	img = np.array(Image.open(img_path))
	height = img.shape[0]
	width = img.shape[1]
	if len(img.shape)!= 3 or img.shape[2] != 3:
		continue

	annotation = map(float,annotation_plt)

	xmin = []
	ymin = []
	xmax = []
	ymax = []
	classes = []
	classes_text = []

	for i in range(len(annotation)/5):

		# class number start from 1
		xmin_ = annotation[5*i+1]-annotation[5*i+3]/2
		xmax_ = annotation[5*i+1]+annotation[5*i+3]/2
		ymin_ = annotation[5*i+2]-annotation[5*i+4]/2
		ymax_ = annotation[5*i+2]+annotation[5*i+4]/2
	        xmax_ = min(xmax_,1)
	        ymax_ = min(ymax_,1)
		xmin.append(xmin_)
		xmax.append(xmax_)
		ymin.append(ymin_)
		ymax.append(ymax_)
		classes.append(int(annotation[5*i])+1)
		classes_text.append(all_categ[classes[i]])

	example = tf.train.Example(features=tf.train.Features(feature={
		'image/height': _int64_feature(height),
		'image/width': _int64_feature(width),
		'image/filename': _byte_feature(filename.encode('utf8')),
		'image/source_id': _byte_feature(filename.encode('utf8')),
		'image/key/sha256': _byte_feature(key.encode('utf8')),
		'image/encoded': _byte_feature(encoded_jpg),
		'iamge/format': _byte_feature('jpeg'.encode('utf8')),
		'image/object/bbox/xmin': _float_list_feature(xmin),
		'image/object/bbox/xmax': _float_list_feature(xmax),
		'image/object/bbox/ymin': _float_list_feature(ymin),
		'image/object/bbox/ymax': _float_list_feature(ymax),
		'image/object/class/text': _byte_list_feature(classes_text),
		'image/object/class/label': _int64_list_feature(classes),

	}))

	writer.write(example.SerializeToString())

writer.close()


