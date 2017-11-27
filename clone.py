import csv
import cv2
import numpy as np
import os

lines =[]
with open('../data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []

#Remove first line
lines.pop(0)

total_datasize = len(lines)
print(total_datasize)
for line in lines:
	source_path = line[0]
	filename = source_path.split('/')[-1]
	current_path = '../data/IMG/' + filename
	if(os.path.exists(current_path)):
		image = cv2.imread(current_path)
		images.append(image)
		measurement = float(line[3])
		measurements.append(measurement)
	else:
		print(current_path, ' does not exist')

X_train = np.asarray(images)
y_train = np.array(measurements)

datasize = X_train.shape[0]
shape_image = X_train[0].shape
nrows, ncols, nchannels = shape_image
shape_image = (datasize,) + shape_image
X_train = np.concatenate(X_train).reshape(shape_image)
print(X_train.shape)

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers import BatchNormalization
from keras.layers.core import Dense, Flatten, Lambda

model = Sequential()
model.add(Lambda(lambda x: x /255.0 - 0.5, input_shape = (nrows,ncols,3)))
model.add(Cropping2D(cropping=((70,25),(1,1)), input_shape=(nrows,ncols,3)))
model.add(BatchNormalization(epsilon=0.001,mode=2, axis=1,input_shape=(nrows-95,ncols-2,3)))
model.add(Convolution2D(24,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
model.add(Convolution2D(36,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
model.add(Convolution2D(48,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
model.add(Flatten())
model.add(Dense(1164, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='tanh'))

#model.compile(loss = 'mse', optimizer='adam')
#model.fit(X_train, y_train, validation_split= 0.2, shuffle=True, nb_epoch = 5)

#model.save('model.h5')