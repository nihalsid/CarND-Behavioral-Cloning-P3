import os
import cv2
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D
import numpy as np

base_input_path = "resources/data/"

dataset_features = []
dataset_label_steering = []

with open(os.path.join(base_input_path, "driving_log.csv"),"r") as fin:
    fin.readline()
    for line in fin.readlines():
        split_line = line.split(",")
        img_center = cv2.imread(os.path.join(base_input_path,split_line[0]))
        img_left = cv2.imread(os.path.join(base_input_path,split_line[0]))
        img_right = cv2.imread(os.path.join(base_input_path,split_line[0]))
        steering = float(split_line[3])
        dataset_features.append(img_center)
        dataset_label_steering.append(steering)

X_train = np.array(dataset_features)
y_train = np.array(dataset_label_steering)

print("Features shape: ",X_train.shape)
print("Labels shape: ",y_train.shape)

# create model
model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Conv2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(16, 5, 5, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)
model.save('model.h5')
