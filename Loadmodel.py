import keras.preprocessing.image
from keras.models import load_model
import sys
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from PIL import Image

from glob import glob
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
class_name = ['cave', 'promenade', 'stairs']
cave = []
promenade = []
stairs = []
newmodel = load_model('./model/05-0.7067.hdf5')
test_x = []
image_datas = glob('./test/*.jpg')
for test_image in image_datas:
    test_img = Image.open(test_image).convert('L')
    test_img = test_img.resize((128, 128))
    test_img = np.array(test_img)
    test_x.append(test_img)
    plt.imshow(test_img, cmap='Greys')
    plt.show()
test_x = np.array(test_x)
test_x = test_x.reshape(test_x.shape[0], 128, 128, 1)

predictions = newmodel.predict(test_x)

print(len(predictions))
for i in range(len(predictions)):
    score = tf.nn.softmax(predictions[i])
    print("{:.2f}perscent".format(100 * np.max(score)), class_name[np.argmax(score)])
    if class_name[np.argmax(score)] == 'cave':
        cave.append(100 * np.max(score))
    if class_name[np.argmax(score)] == 'promenade':
        promenade.append(100 * np.max(score))
    if class_name[np.argmax(score)] == 'stairs':
        stairs.append(100 * np.max(score))

print(len(cave))
print(len(promenade))
print(len(stairs))