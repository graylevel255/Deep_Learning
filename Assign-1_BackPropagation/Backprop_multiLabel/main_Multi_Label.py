# -*- coding: utf-8 -*-

import Backprop_MultiLabel as bp
import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score



def normalize(x, x_min, x_max):
    # perform min max normalization on input data
    return (x-x_min)/(x_max.astype(float) - x_min.astype(float))


def get_precision(y_desired, y_predicted):
    n = len(y_desired)
    true_pos = [y_d * y_p for y_d, y_p in zip(y_desired, y_predicted)]
    true_pos = np.sum(true_pos, axis=1)
    total_pos =  np.sum(y_predicted, axis=1)
    indices = total_pos == 0
    total_pos[indices] = 1.0
    precision = np.array(true_pos) / total_pos
    precision[indices] = 0.0

    return (1.0/n)*np.sum(precision)


def get_recall(y_desired, y_predicted):
    n = len(y_desired)
    true_pos = [y_d * y_p for y_d, y_p in zip(y_desired, y_predicted)]
    true_pos = np.sum(true_pos, axis=1)
    return (np.sum(np.array(true_pos)/np.array(np.sum(y_desired, axis=1, dtype = np.float))))/n


def get_f_measure(precision, recall):
    return 2.0 * ((precision * recall) / (precision + recall))

# Read the input data
dirname = os.path.dirname(__file__)
in_train_file = dirname + '/train_X.txt'
out_train_file = dirname + '/train_Y.txt'

in_test_file = dirname + '/test_X.txt'
out_test_file = dirname + '/test_Y.txt'

ip_train_X = np.loadtxt(in_train_file)
#ip_train_X = ip_train_X - np.mean(ip_train_X, axis=0)
op_train_Y = np.loadtxt(out_train_file)

print(np.sum(op_train_Y, axis=0))
ip_test_X = np.loadtxt(in_test_file)
#ip_test_X = ip_test_X - np.mean(ip_train_X, axis=0)
op_test_Y = np.loadtxt(out_test_file)

n_train = len(ip_train_X)
n_test = len(ip_test_X)

print n_train
print n_test

# train the model for Batch update, delta rule, normalized features
# normalize the data



# Normalizing training data

x_max = np.max(ip_train_X, axis=0)
x_min = np.min(ip_train_X, axis=0)

#ip_train_X = normalize(ip_train_X, x_min, x_max)
#ip_test_X = normalize(ip_test_X, x_min, x_max)

# ip_train = ip_list[:n_train]
# op_train = op_list[:n_train]

# ip_test = ip_list[n_train:]
# op_test = op_list[n_train:]

## write split data into a train and test file

nn = bp.NeuralNet([32, 50, 45, 6])
# storing initial weights
with open(dirname + '/initial_weights.txt', 'w') as f:
    for item in np.array(nn.weights):
            f.write("%s " % item)
            f.write("\n")

eta = .01
#threshold = .0001
error_old = 1000
epochs = 0
while True:
    nn.update_weights_batch(ip_train_X, op_train_Y, n_train, eta, "adam")
    if epochs % 20 == 0:
        predicted, error = nn.test(ip_train_X, op_train_Y)
        print (epochs, ':', error)
        if np.abs(error_old - error) < 1e-3:
         break
        error_old = error
    epochs += 1

print(epochs)
print (nn.get_weights())
print(nn.get_bias())

# Calculate Mean Squared Error on test data
y, mse = nn.test(ip_test_X, op_test_Y)
y = np.array(y)
y[y >= 0.50] = 1
y[y < 0.50] = 0

#pres_test = precision_score(op_test_Y, y, labels=None, pos_label=1, average=None, sample_weight=None)
#print(pres_test)
print("test precision", get_precision(op_test_Y, y))
print("test Recall", get_recall(op_test_Y, y))
print("test F measure", get_f_measure(get_precision(op_test_Y, y), get_recall(op_test_Y, y)))

print(mse)
# print np.array(op_test_Y)

with open(dirname+'/test_predicted_Labels.txt', 'w') as f:
    for item in np.array(y):
        for value in item:
            f.write("%s " % value)
        f.write("\n")

# Calculate Mean Squared Error on train data
y, mse = nn.test(ip_train_X, op_train_Y)

with open(dirname+'/train_predicted_prob.txt', 'w') as f:
    for item in np.array(y):
        for value in item:
            f.write("%s " % value)
        f.write("\n")

y = np.array(y)
y[y >= 0.50] = 1
y[y < 0.50] = 0


with open(dirname+'/train_predicted.txt', 'w') as f:
    for item in np.array(y):
        for value in item:
            f.write("%s " % value)
        f.write("\n")

#pres_test = precision_score(op_train_Y, y, labels=None, pos_label=1, average = None, sample_weight=None)
#print(pres_test)

print("train Precision", get_precision(op_train_Y, y))
print("train Recall",get_recall(op_train_Y, y))
print ("train FMeasure", get_f_measure(get_precision(op_train_Y, y), get_recall(op_train_Y, y)))