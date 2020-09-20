import tensorflow as tf
import csv
import numpy as np
import math
import sklearn.model_selection as sk
import config

# def next_batchmlffnn(batch_num):
#     x_batch = data[batch_num*batch_size: (batch_num+1)*batch_size, :]
#     y_batch = labels[batch_num*batch_size: (batch_num+1)*batch_size]
#     return x_batch, y_batch

def next_batch(batch_num):
    batch = data[batch_num*batch_size: (batch_num+1)*batch_size, :]
    return batch


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

with open('./ip2ae3.csv', 'r') as csvfile:
    ip2ae3 = list(csv.reader(csvfile))


features = np.array(ip2ae3)
data, test_data = sk.train_test_split(features, test_size=0.2, random_state=9)
data = np.array(data, dtype=float)
test_data = np.array(test_data, dtype=float)
batch_size = config.ae1batch
n_batches = int(math.ceil(np.array(data).shape[0]/batch_size))

# initialize inputs
x = tf.placeholder(tf.float32, shape=[None, np.array(data).shape[1]], name='x')

# ******************** BUILDING THE MODEL ******************** #
n_neurons1 = config.ae3h1
n_neurons2 = config.ae3h2

# Weights and bais initialization

initializer = tf.contrib.layers.xavier_initializer(seed=1)
W1 = tf.Variable(initializer([np.array(data).shape[1], n_neurons1]))
b1 =tf.Variable(tf.constant(0.01, shape=[n_neurons1]))

initializer = tf.contrib.layers.xavier_initializer(seed=65)
W2 = tf.Variable(initializer([n_neurons1, n_neurons2]))
b2 =tf.Variable(tf.constant(0.01, shape=[n_neurons2]))

initializer = tf.contrib.layers.xavier_initializer(seed=90)
W3 = tf.Variable(initializer([n_neurons2, n_neurons1]))
b3 =tf.Variable(tf.constant(0.01, shape=[n_neurons1]))

initializer = tf.contrib.layers.xavier_initializer(seed=98)
W4 = tf.Variable(initializer([n_neurons1, np.array(data).shape[1]]))
b4 =tf.Variable(tf.constant(0.01, shape=[np.array(data).shape[1]]))

# Apply activations
l1 = tf.nn.sigmoid(tf.math.add(tf.matmul(x, W1), b1))
l2 = tf.math.add(tf.matmul(l1, W2), b2)
l3 = tf.nn.sigmoid(tf.math.add(tf.matmul(l2, W3), b3))
out = tf.math.add(tf.matmul(l3, W4), b4)

# Defining the loss
loss = tf.reduce_mean(tf.squared_difference(x, out))

# initializing the optimizer
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

init = tf.global_variables_initializer()
ip2ae2 = []
w3_right = []
b3_right = []
w4_left = []
b4_left = []

# ************** TRAINING MODEL ******************#
with tf.Session() as sess:
    sess.run(init)
    current_epoch_loss = 500.0
    previous_epoch_loss = 1000.0
    epochs = 0.0
    # while previous_epoch_loss - current_epoch_loss > config.ae1threshold:
    for i in range(config.ae3epoch):
        previous_epoch_loss = current_epoch_loss
        epoch_loss = 0.0
        for i in range(n_batches):
            batch = next_batch(i)
            summary = sess.run(train_step, feed_dict={x: batch})
        current_epoch_loss = sess.run(loss, feed_dict={x: data})
        validation_loss = sess.run(loss, feed_dict={x: test_data})
        print("Step : %d. Train_loss: %g. Validation_loss: %g" %(epochs, current_epoch_loss, validation_loss))
        epochs += 1
    # ip2mlffnn = sess.run(l2, feed_dict={x: features})
    w3_right = sess.run(W1)
    b3_right = sess.run(b1)
    w4_left = sess.run(W2)
    b4_left = sess.run(b2)
    with open('w3_right.csv', 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(w3_right)
    writeFile.close()

    with open('b3_right.csv', 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerow(b3_right)
    writeFile.close()

    with open('w4_left.csv', 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(w4_left)
    writeFile.close()

    with open('b4_left.csv', 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerow(b4_left)
    writeFile.close()
print("OPTIMIZATION OF AE3 FINISHED")
