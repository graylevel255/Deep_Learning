from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import numpy as np
import csv
import os
import fileinput
import re

base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
features = []

# files = os.listdir('.\Image_Caption_data\\6\\small_train\\')
filename = '.\Image_Caption_data\\6\\small_data.txt'

with open(filename) as file:
    content = file.readlines()
    fil = [re.findall('.*jpg', line) for line in content]
for f in fil:
    # #  img_path = 'image/' + row[0].replace('\\', '/')
    img_path = '.\Image_Caption_data\\6\\small_data\\data\\' + f[0]
    #img_path='./Image_Caption_data/6/small_train/' + f
    print("Loading Image " + img_path)
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features.append(model.predict(img_data).reshape(4096))

features = np.array(features)
print(features.shape)
np.save('feats_train.npy', features)

