from __future__ import print_function, division
import os
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
import csv
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


#  Read the input data
# dirname = os.path.dirname(__file__)
# input_data = dirname + '/X_redhl.txt'
# ip_file = np.loadtxt(input_data)
# dim = ip_file.shape[1]
# input_data = np.array(ip_file[:, 0:dim-1])
# output_data = np.array(ip_file[:, dim-1:])
# print(ip_file.shape)
# print(input_data.shape)
# print(output_data.shape)


dirname = os.path.dirname(__file__)
X_train_file = dirname + '/X_red_train.txt'

X_train = np.array(np.loadtxt(X_train_file), dtype=float)
with open('./label_unsupervised.csv', 'r') as csvFile:
    y_train = list(csv.reader(csvFile))
y_train = np.array(y_train, dtype=float).T

dim = X_train.shape[1]


# print(ip_file.shape)
# print(input_data.shape)
# print(output_data.shape)
# Randomly divide 70% for train and 20% for test, 10% for validation
initial_split = .2
X_remain_file = dirname + '/X_red_remain.txt'
X_remain = np.array(np.loadtxt(X_remain_file))
with open('./label_supervised.csv', 'r') as csvFile:
    y_remain = list(csv.reader(csvFile))
y_remain = np.array(y_remain, dtype=float).T


X_val, X_test, y_val, y_test = train_test_split(X_remain, y_remain, test_size=.5, random_state=4)

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

oneOfK_train = np.array(oneOfK_train)
oneOfK_test = np.array(oneOfK_test)
threshold = .01

def ceil(a, b):
    return -(-a//b)


def next_batch(batch_num):
    batch_x = X_train[batch_num*batch_size: (batch_num+1)*batch_size, :]
    batch_y = oneOfK_train[batch_num*batch_size: (batch_num+1)*batch_size, :]
    return batch_x, batch_y

# Parameters
learning_rate = 0.0001
batch_size = 16


# Network Parameters
n_hidden_1 = 50  # 1st layer number of neurons
n_hidden_2 = 30    #t 2nd layer number of neurons
num_input = dim  #  number of nodes in input layer
num_classes = 5    #  total classes (0-4 )

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])
keep_prob = tf.constant(.5)
epsilon = 1e-3
beta =0.01

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.truncated_normal([num_input, n_hidden_1], seed=1)*.01),
    'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], seed=2)*.01),
    'out': tf.Variable(tf.truncated_normal([n_hidden_2, num_classes], seed=3)*.01)
}
biases = {
    'b1': tf.Variable(tf.truncated_normal([n_hidden_1], seed=4)*0.01),
    'b2': tf.Variable(tf.truncated_normal([n_hidden_2], seed=5)*0.01),
    'out': tf.Variable(tf.truncated_normal([num_classes], seed=6)*0.01)
}

# Create model
def neural_net(x, keep_prob):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.nn.leaky_relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    # apply DropOut to hidden layer 1
    drop_out_1 = tf.nn.dropout(layer_1, keep_prob)
    # Hidden fully connected layer with 256 neurons
    # z_BN2 = tf.matmul(drop_out_1, weights['h2'])
    # batch_mean2, batch_var2 = tf.nn.moments(z_BN2, [0])
    # scale2 = tf.Variable(tf.ones([n_hidden_2]))
    # beta2 = tf.Variable(tf.zeros([n_hidden_2]))
    # BN2 = tf.nn.batch_normalization(z_BN2, batch_mean2, batch_var2, beta2, scale2, epsilon)
    # layer_2 = tf.nn.leaky_relu(BN2)
    layer_2 = tf.nn.leaky_relu(tf.add(tf.matmul(drop_out_1, weights['h2']), biases['b2']))
    # apply DropOut to hidden layer 2
    drop_out_2 = tf.nn.dropout(layer_2, keep_prob)
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(drop_out_2, weights['out']) + biases['out']
    # out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


# Construct model
logits = neural_net(X, keep_prob)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))

# L2 loss
#reg = tf.nn.l2_loss(weights['h1']) + tf.nn.l2_loss(weights['h2'])
#loss = tf.reduce_mean(loss_op + reg * beta)
#loss = tf.reduce_mean(loss_op)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

num_batches_train = ceil(n_train, batch_size)
# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    current_epoch_loss = 500.0
    previous_epoch_loss = 1000.0
    current_epoch_loss_val = 500.0
    previous_epoch_loss_val = 1000.0
    epochs = 0

    while  current_epoch_loss > 1: # and previous_epoch_loss_val - current_epoch_loss_val > .001:
        previous_epoch_loss = current_epoch_loss
        current_epoch_loss = 0
        current_epoch_loss_val = previous_epoch_loss_val
        current_epoch_loss_val = 0

        for j in range(0, num_batches_train):
            batch_x, batch_y = next_batch(j)
            summary, train_loss = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})
            # current_epoch_loss += train_loss

        current_epoch_loss = sess.run(loss_op, feed_dict={X: X_train, Y: oneOfK_train})
        current_epoch_loss_val = sess.run(loss_op, feed_dict={X: X_val, Y: oneOfK_val})

        print(" Epochs: %d. Loss = %g, val_loss: %g" % (epochs, current_epoch_loss, current_epoch_loss_val))
        epochs = epochs + 1

    print("Optimization Finished!")

    # Calculate accuracy for  train images
    print("Train Accuracy:", \
          sess.run(accuracy, feed_dict={X: X_train, Y: oneOfK_train}))
    logits_tr = sess.run(logits, feed_dict={X: X_train, Y: oneOfK_train})
    y_predict = tf.argmax(logits_tr, 1)
    confusion = tf.confusion_matrix(labels=y_train, predictions=y_predict, num_classes=num_classes)
    print(sess.run(confusion))

    # Calculate accuracy for  validation images
    print("Validation Accuracy:", \
          sess.run(accuracy, feed_dict={X: X_val, Y: oneOfK_val}))

    # Calculate accuracy for  test images
    print("Testing Accuracy:", \
          sess.run(accuracy, feed_dict={X: X_test, Y: oneOfK_test}))

    logits = sess.run(logits, feed_dict={X: X_test, Y: oneOfK_test})
    y_predict = tf.argmax(logits, 1)
    confusion = tf.confusion_matrix(labels=y_test, predictions=y_predict, num_classes=num_classes)
    print(sess.run(confusion))

