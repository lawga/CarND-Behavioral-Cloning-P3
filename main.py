import csv
import pickle

import cv2
import numpy as np

from scipy import ndimage
from tqdm import tqdm

lines = []

with open('../data/P3/driving_log_raw.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurments = []

new = 1
correction_factor = 0.2

if new:
    for line in tqdm(lines):
        source_path_center = line[0]
        source_path_left = line[1]
        source_path_right = line[2]
        filename_center = source_path_center.split('/')[-1]
        filename_left = source_path_left.split('/')[-1]
        filename_right = source_path_right.split('/')[-1]
        current_path_center = '../data/P3/IMG/' + filename_center
        current_path_left = '../data/P3/IMG/' + filename_left
        current_path_right = '../data/P3/IMG/' + filename_right
        image_center = ndimage.imread(current_path_center)
        image_left = ndimage.imread(current_path_left)
        image_right = ndimage.imread(current_path_right)
        images.append(image_center)
        images.append(image_left)
        images.append(image_right)
        images.append(cv2.flip(image_center,1))
        images.append(cv2.flip(image_left,1))
        images.append(cv2.flip(image_right,1))
        measurment_center = float(line[3])
        measurments.append(measurment_center) #image center
        measurments.append((measurment_center+correction_factor)) #image left
        measurments.append((measurment_center-correction_factor)) #image right
        measurments.append(measurment_center*-1.0) #image center flipped
        measurments.append((measurment_center+correction_factor)*-1.0) #image left flipped
        measurments.append((measurment_center-correction_factor)*-1.0) #image right flipped

    X_train = np.array(images)
    y_train = np.array(measurments)

    # X_train_param = X_train
    # X_train_data = '../data/P3/images_data/X_train.pickle'
    # with open(X_train_data, 'wb') as dump_file:
    #     pickle.dump(X_train_param, dump_file)

    # y_train_param = y_train
    # y_train_data = '../data/P3/images_data/y_train.pickle'
    # with open(y_train_data, 'wb') as dump_file:
    #     pickle.dump(y_train_param, dump_file)

if not(new):
    X_train_data = '../data/P3/images_data/X_train.pickle'
    with open(X_train_data, 'rb') as dump_file:
        X_train = pickle.load(dump_file)

    y_train_data = '../data/P3/images_data/y_train.pickle'
    with open(y_train_data, 'rb') as dump_file:
        y_train = pickle.load(dump_file)

from keras.layers import Dense, Flatten, Lambda
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Convolution2D(6, 5, 5, activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6, 5, 5, activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train,
          y_train,
          validation_split=0.2,
          shuffle=True,
          nb_epoch=5)

model.save('model.h5')
