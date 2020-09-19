# Save split data of train-test
import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split


# Read the input data
dirname = os.path.dirname(__file__)
input_data = dirname + '/energy-data.txt'
ip_file = np.loadtxt(input_data)


# Randomly divide 70% for train and 30% for test
train, test = train_test_split(ip_file, test_size=0.3)

with open('train.txt', 'w') as f:
    for item in np.array(train):
        for value in item:
            f.write("%s " % value)
        f.write("\n")

with open('test.txt', 'w') as f:
    for item in np.array(test):
        for value in item:
            f.write("%s " % value)
        f.write("\n")