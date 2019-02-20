import csv
import os
import pickle
from random import shuffle


import cv2
import numpy as np
import sklearn

from scipy import ndimage
from tqdm import tqdm

from keras.models import Model
import matplotlib.pyplot as plt


lines = []

with open('../data/P3/driving_log_raw.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)


from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

new = 1
correction_factor = 0.2

images = []
measurments = []

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurments = []
            for batch_sample in batch_samples:
                source_path_center = batch_sample[0]
                source_path_left = batch_sample[1]
                source_path_right = batch_sample[2]
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
                measurment_center = float(batch_sample[3])
                measurments.append(measurment_center) #image center
                measurments.append((measurment_center+correction_factor)) #image left
                measurments.append((measurment_center-correction_factor)) #image right
                measurments.append(measurment_center*-1.0) #image center flipped
                measurments.append((measurment_center+correction_factor)*-1.0) #image left flipped
                measurments.append((measurment_center-correction_factor)*-1.0) #image right flipped

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(measurments)
            yield sklearn.utils.shuffle(X_train, y_train)

if 1:
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

    #OverflowError: cannot serialize a bytes object larger than 4 GiB
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

# compile and train the model using the generator function
# train_generator = generator(train_samples, batch_size=32)
# validation_generator = generator(validation_samples, batch_size=32)

from keras.layers import Dense, Flatten, Lambda
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Cropping2D, Conv2D

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((75,25), (0,0)), input_shape=(3,160,320)))
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Conv2D(64, (2, 2), activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

# history_object = model.fit_generator(train_generator, steps_per_epoch= len(train_samples)*6,
# validation_data=validation_generator, validation_steps=len(validation_samples)*6, epochs=1, verbose = 1)
history_object = model.fit(X_train,
          y_train,
          validation_split=0.2,
          shuffle=True,
          nb_epoch=3,
          verbose = 1)



model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epochP5325
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()