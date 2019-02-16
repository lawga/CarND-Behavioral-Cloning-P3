import csv
import pickle

import cv2
import numpy as np
from keras.layers import Dense, Flatten, Lambda
from keras.models import Sequential
from scipy import ndimage
from tqdm import tqdm

lines = []

with open('../data/P3/driving_log_raw.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurments = []

if 0:
    for line in tqdm(lines):
        source_path = line[0]
        filename = source_path.split('/')[-1]
        current_path = '../data/P3/IMG/' + filename
        image = ndimage.imread(current_path)
        images.append(image)
        measurment = float(line[3])
        measurments.append(measurment)

    X_train = np.array(images)
    y_train = np.array(measurments)

    X_train_param = X_train
    X_train_data = '../data/P3/images_data/X_train.pickle'
    with open(X_train_data, 'wb') as dump_file:
        pickle.dump(X_train_param, dump_file)

    y_train_param = y_train
    y_train_data = '../data/P3/images_data/y_train.pickle'
    with open(y_train_data, 'wb') as dump_file:
        pickle.dump(y_train_param, dump_file)

X_train_data = '../data/P3/images_data/X_train.pickle'
with open(X_train_data, 'rb') as dump_file:
    X_train = pickle.load(dump_file)

y_train_data = '../data/P3/images_data/y_train.pickle'
with open(y_train_data, 'rb') as dump_file:
    y_train = pickle.load(dump_file)


model = Sequential()
model.add(Lambda(lambda x: x/255.0, input_shape=(160, 320, 3)))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train,
          y_train,
          validation_split=0.2,
          shuffle=True,
          nb_epoch=4)

model.save('model.h5')
