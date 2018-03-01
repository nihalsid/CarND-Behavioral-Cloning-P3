import os
import cv2
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D, Cropping2D
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split 

base_input_path = "resources/data/"
dataset = []


with open(os.path.join(base_input_path, "driving_log.csv"),"r") as fin:
	fin.readline()
	for line in fin.readlines():
		split_line = line.split(",")
		img_center = os.path.join(base_input_path,split_line[0])
		img_left = os.path.join(base_input_path,split_line[1].strip())
		img_right = os.path.join(base_input_path,split_line[2].strip())
		steering = float(split_line[3])
		dataset.append([img_center, img_left, img_right, steering])

		
def generator(samples, batch_size):
	while 1:
		shuffle(samples)
		for i in range(int(len(samples)/batch_size)):
			features = []
			label_steering = []
			for j in range(batch_size):	
				sample = samples[i*batch_size+j]
				features.append(cv2.imread(sample[0]))
				features.append(cv2.flip(cv2.imread(sample[0]),1))
				features.append(cv2.imread(sample[1]))
				features.append(cv2.imread(sample[2]))
				label_steering.append(sample[3])
				label_steering.append(sample[3]*-1)
				label_steering.append(sample[3]+0.2)
				label_steering.append(sample[3]-0.2)
			X_train = np.array(features)
			y_train = np.array(label_steering)
			yield shuffle(X_train, y_train)

train_samples, validation_samples = train_test_split(dataset, test_size=0.2)
train_generator = generator(train_samples, 32)
validation_generator = generator(validation_samples, 32)

# create model
model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70, 25),(0, 0))))
model.add(Conv2D(24, 5, 5, subsample=(2,2), activation='relu'))
model.add(Conv2D(36, 5, 5, subsample=(2,2), activation='relu'))
model.add(Conv2D(48, 5, 5, subsample=(2,2), activation='relu'))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*4, validation_data=validation_generator, nb_val_samples=len(validation_samples)*4, nb_epoch=5)
model.save('model.h5')
