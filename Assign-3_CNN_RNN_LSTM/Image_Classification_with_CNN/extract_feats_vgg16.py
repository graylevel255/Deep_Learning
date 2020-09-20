from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import numpy as np
from sklearn.model_selection import train_test_split
import glob
import os

features = []
# i = 0
# data_directory = '/home/ubantu/Desktop/DL_A3 Data/classification task/my_class_data/'
# directories = [d for d in os.listdir(data_directory)
#             if os.path.isdir(os.path.join(data_directory, d))]
# print(directories)
#
# for d in directories:
#     for image_file in glob.iglob(data_directory + d + '/*.jpg'):
#         print(i)
#         i += 1


# base_model = VGG16(weights='imagenet')
# model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
#
# # for i in range(images)
#
# chotadata = '/home/ubantu/Desktop/DL_A3 Data/chota_data/'
#
#
# directories = [d for d in os.listdir(chotadata)
#              if os.path.isdir(os.path.join(chotadata, d))]
# i = 0
# # for d in directories:
# #     for image_file in glob.iglob(directories + d + '/*.jpg'):
# #         print(i)
# #         i += 1
#
# for d in directories:
#     for image_file in glob.iglob(chotadata + d + '/*.jpg'):
#         img = image.load_img(image_file, target_size=(224, 224))
#         img_data = image.img_to_array(img)
#         img_data = np.expand_dims(img_data, axis=0)
#         img_data = preprocess_input(img_data)
#         feats = model.predict(img_data)
#         feats = feats.reshape(feats.shape[1])
#         features.append(feats)
#
# vgg16_features = np.array(features)
# print(vgg16_features.shape)
# print(type(vgg16_features))
# print(vgg16_features)
#
# # Save features in a csv file
#
# dirname = os.path.dirname(__file__)
# with open(dirname+'/mini_features.txt', 'w') as f:
#     for item in np.array(vgg16_features):
#         for value in item:
#             f.write("%s " % value)
#         f.write("\n")
# print("Features Extracted Successfully\n")

# Read features :

dirname = os.path.dirname(__file__)
feats = dirname + '/InceptionV3_features.txt'

data = np.array(np.loadtxt(feats), dtype=float)
# with open('./label_unsupervised.csv', 'r') as csvFile:
#     y_train = list(csv.reader(csvFile))
# y_train = np.array(y_train, dtype=float).T

dim = data.shape[1]
print(dirname)
print(data.shape, dim)

# create one hot vector with target vector
N = data.shape[0]     # total egs
n1 = 270
n2 = 798
n3 = 209
n4 = 201
n5 = 800
oneOfK = [np.zeros(5) for i in range(N)]

n11 = 1/n1

# create target vector
y = np.zeros(N)
for i in range(n1):
    y[i] = 0
for i in range(n2):
    y[i+n1] = 1
t1 = n1 + n2
for i in range(n3):
    y[i+t1] = 2
t2 = t1 + n3
for i in range(n4):
    y[i+t2] = 3
t3 = t2 + n4
for i in range(n5):
    y[i+t3] = 4
print(y.shape)
for i in range(N):
    oneOfK[i][int(y[i])] = 1
print(oneOfK)
# Split into train test and val


X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=.3, random_state=42)

X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.33, random_state=51)

with open(dirname+'/X_traini.txt', 'w') as f:
    for item in np.array(X_train):
        for value in item:
            f.write("%s " % value)
        f.write("\n")

with open(dirname+'/Y_traini.txt', 'w') as f:
    for item in np.array(y_train):
        f.write("%f " % item)
        f.write("\n")

with open(dirname+'/X_vali.txt', 'w') as f:
    for item in np.array(X_val):
        for value in item:
            f.write("%s " % value)
        f.write("\n")

with open(dirname+'/Y_vali.txt', 'w') as f:
    for item in np.array(y_val):
        f.write("%f " % item)
        f.write("\n")

with open(dirname+'/X_testi.txt', 'w') as f:
    for item in np.array(X_test):
        for value in item:
            f.write("%s " % value)
        f.write("\n")

with open(dirname+'/Y_testi.txt', 'w') as f:
    for item in np.array(y_test):
        f.write("%f " % item)
        f.write("\n")

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, X_val.shape, y_val.shape)

n_train = X_train.shape[0]
n_val = X_val.shape[0]
n_test = X_test.shape[0]

oneOfK_train = [np.zeros(5) for i in range(n_train)]
oneOfK_val = [np.zeros(5) for i in range(n_val)]
oneOfK_test = [np.zeros(5) for i in range(n_test)]

for i in range(n_train):
    oneOfK_train[i][int(y_train[i])] = 1

for i in range(n_test):
    oneOfK_test[i][int(y_test[i])] = 1

for i in range(n_val):
    oneOfK_val[i][int(y_val[i])] = 1


print(oneOfK_train)
print("TRAIN DONE\n")
print(oneOfK_test)
print("TEST DONE\n")
print(oneOfK_val)

print(sum(y_train == 3))