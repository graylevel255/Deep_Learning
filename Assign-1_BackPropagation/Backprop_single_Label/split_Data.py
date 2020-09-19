# Save split data of train-test
import os
import numpy as np
from sklearn.model_selection import train_test_split


# Read the input data
dirname = os.path.dirname(__file__)
input_data = dirname + '/total_data.txt'
ip_file = np.loadtxt(input_data)
data = ip_file
# Randomly divide 70% for train and 30% for test
train_data, test_data = train_test_split(ip_file, test_size = 0.3)
# train_data = data[0:1254, :]
# test_data = data[1254:, :]

with open('train.txt', 'w') as f:
    for item in np.array(train_data):
        for value in item:
            f.write("%s " % value)
        f.write("\n")

with open('test.txt', 'w') as f:
    for item in np.array(test_data):
        for value in item:
            f.write("%s " % value)
        f.write("\n")