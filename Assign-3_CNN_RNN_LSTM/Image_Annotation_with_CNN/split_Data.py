# Save split data of train-test
import os
import numpy as np
import csv
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split


# Read the input data
with open('features_VGG16.csv', newline='') as csvfile:
    data_train = list(csv.reader(csvfile))

with open('Y_clean_small.csv', newline='') as csvfile:
    Y_train = list(csv.reader(csvfile))

data_train = np.array(data_train)
Y_train = np.array(Y_train)

# Randomly divide 70% for train and 30% for test
train_x, remain_x, train_y, remain_y = train_test_split(data_train, Y_train, test_size=0.3)
val_x, test_x, val_y, test_y = train_test_split(remain_x, remain_y, test_size=.65)

with open('train_x_VGG.txt', 'w') as f:
    for item in np.array(train_x):
        for value in item:
            f.write("%s " % value)
        f.write("\n")

with open('test_x_VGG.txt', 'w') as f:
    for item in np.array(test_x):
        for value in item:
            f.write("%s " % value)
        f.write("\n")

with open('val_x_VGG.txt', 'w') as f:
    for item in np.array(val_x):
        for value in item:
            f.write("%s " % value)
        f.write("\n")

with open('train_Y_VGG.txt', 'w') as f:
    for item in np.array(train_y):
        for value in item:
            f.write("%s " % value)
        f.write("\n")

with open('test_Y_VGG.txt', 'w') as f:
    for item in np.array(test_y):
        for value in item:
            f.write("%s " % value)
        f.write("\n")

with open('val_Y_VGG.txt', 'w') as f:
    for item in np.array(val_y):
        for value in item:
            f.write("%s " % value)
        f.write("\n")

