import csv
import cv2
import numpy as np
import os

#Define image size
nrows = 160
ncols = 320

#Function: get_data: Gets data as a batch from a whole data set. 
#	This function reads the center, left and right images and create steering measurements for each.
#	This also creates a flipped image for each center, left and right images and corresponding negative measurements.
#	Hence for each line, 6 data are obtained.
#	Final length of output is 6 x batch_size
#Input: last_start: Index where last batch ended.
#		lines: lines of text file that has information about images and steering angles
#		rand_indeces: randomized indeces.
#		batch_size: batch size
#		data_size: Total length of data size	
#Output: Numpy array of image data, steering measurements and Index where current batch ended 
def get_data(last_start, lines, rand_indeces, batch_size, data_size):
	stop = last_start + batch_size
	if stop > data_size:
		stop = data_size
	
	images = []
	measurements = []
	correction = np.array([0.0, 0.2, -0.2]) #center, left, right
	#print('\n',last_start,stop)
	for i in range(last_start,stop):
		for j in range(0,3):
			source_path = lines[rand_indeces[i]][j]
			filename = source_path.split('/')[-1]
			current_path = '../data/IMG/' + filename
			if(os.path.exists(current_path)):
				image = cv2.imread(current_path)
				images.append(image)
				measurement = float(lines[rand_indeces[i]][3]) + correction[j]
				measurements.append(measurement)
				#Flipped image
				image_flipped = np.fliplr(image)
				images.append(image_flipped)
				measurement_flipped = -measurement
				measurements.append(measurement_flipped)
			else:
				print(current_path, ' does not exist')
			
	X = np.asarray(images)
	y = np.array(measurements)
	datasize = X.shape[0]
	shape_image = (datasize,nrows,ncols,3)
	X = np.concatenate(X).reshape(shape_image)
	return X, y, stop
	
#Generator: get_train_data_from_generator: Yields training data as a batch from a whole data set. 	
#Input: lines: lines of text file that has information about images and steering angles
#		rand_indeces: randomized indeces.
#		partition_ind: Index where training data ends in whole data set
#		batch_size: batch size
#		train_data_size: Total length of training data size	
def get_train_data_from_generator(lines, rand_indeces, partition_ind, batch_size, train_data_size):
	last_start_train = 0
	while(True):
		X_train, y_train, last_start_train = get_data(last_start_train, lines, rand_indeces, batch_size, train_data_size)
		if last_start_train == train_data_size:
			last_start_train = 0
		yield (X_train, y_train)

#Generator: get_val_data_from_generator: Yields validation data as a batch from a whole data set. 	
#Input: lines: lines of text file that has information about images and steering angles
#		rand_indeces: randomized indeces.
#		partition_ind: Index where training data ends in whole data set
#		batch_size: batch size
#		total_data_size: Total length of data size	
def get_val_data_from_generator(lines, rand_indeces, partition_ind, batch_size, total_data_size):
	last_start_val = partition_ind
	while(True):
		X_val, y_val, last_start_val = get_data(last_start_val, lines, rand_indeces, batch_size, total_data_size)
		if last_start_val == total_data_size:
			last_start_val = partition_ind
		yield (X_val, y_val)	
				
#Read lines for driving_log
lines =[]
with open('../data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)
#Remove first line
lines.pop(0)

#Shuffle data and split data in training and validation
split = 0.2
total_data_size = len(lines)
rand_indeces = np.arange(total_data_size)
np.random.shuffle(rand_indeces)
partition_ind = int(total_data_size*(1.0-split))
train_data_size = partition_ind
val_data_size = total_data_size - train_data_size

#Define batch size
batch_size = 256 #actual size is 256*6
train_batch_epochs = int(train_data_size/batch_size) + 1
val_batch_epochs = int(val_data_size/batch_size) + 1

#Load Keras based modules
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers import BatchNormalization
from keras.layers.core import Dense, Flatten, Lambda
import keras as ks
		
#Define model. Please refer to NVIDIA self driving pipeline.
model = Sequential()
model.add(Lambda(lambda x: x /255.0 - 0.5, input_shape = (nrows,ncols,3)))
model.add(Cropping2D(cropping=((70,25),(0,0)), input_shape=(nrows,ncols,3)))
model.add(BatchNormalization(epsilon=0.001, axis=1,input_shape=(nrows-95,ncols,3)))
model.add(Convolution2D(24,(5,5),padding='valid', activation='relu', strides=(2,2)))
model.add(Convolution2D(36,(5,5),padding='valid', activation='relu', strides=(2,2)))
model.add(Convolution2D(48,(5,5),padding='valid', activation='relu', strides=(2,2)))
model.add(Convolution2D(64,(3,3),padding='valid', activation='relu', strides=(1,1)))
model.add(Convolution2D(64,(3,3),padding='valid', activation='relu', strides=(1,1)))
model.add(Flatten())
model.add(Dense(1164, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='tanh'))

#Define loss and call fit_generator to train the model.
model.compile(loss = 'mse', optimizer='adam')
model.fit_generator(generator = get_train_data_from_generator(lines, rand_indeces, partition_ind, batch_size, train_data_size), steps_per_epoch = train_batch_epochs, validation_data =  get_val_data_from_generator(lines, rand_indeces, partition_ind, batch_size, total_data_size), validation_steps = val_batch_epochs, epochs = 5, shuffle=True)

#Save model in hdfs format.
model.save('model.h5')