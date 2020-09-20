from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# read the input data

#
# def load_data(data_directory):
#     directories = [d for d in os.listdir(data_directory)
#                    if os.path.isdir(os.path.join(data_directory, d))]
#     labels = []
#     image_features = []
#     label = 0
#     for d in directories:
#         label_directory = os.path.join(data_directory, d)
#         file_names = [os.path.join(label_directory, f)
#                       for f in os.listdir(label_directory) if f.endswith(".txt")]
#
#         for f in file_names:
#             feature = np.loadtxt(f, delimiter=' ')
#             image_features.append(feature.flatten())
#             labels.append(label)
#         label += 1
#     return image_features, labels
#
#
# ROOT_PATH = './SingleLabelImageFeatures/Features/'
#
# image_features, labels = load_data(ROOT_PATH)
# X_train = np.array(image_features)
# y_train = np.array(labels)
# # plt.hist(labels, 10)
# plt.show()
# print(image_features.shape)
# print(labels.shape)
# n = image_features.shape[0]
# Splitting the data and Train, Val and Test
# initial_split = .3
# X_train, X_remain, y_train, y_remain = train_test_split(image_features, labels, test_size=initial_split, random_state=9)
# new_split = 10.0/(initial_split*100.)
# X_val, X_test, y_val, y_test = train_test_split(X_remain, y_remain, test_size=(1-new_split), random_state=4)

with open('./features_unsupervised.csv', 'r') as csvFile:
    X_train = list(csv.reader(csvFile))
X_train = np.array(X_train, dtype=float)
n_train = X_train.shape[0]


with open('./features_supervised.csv', 'r') as csvFile:
    X_remain = list(csv.reader(csvFile))
X_remain = np.array(X_remain, dtype=float)

# Making mini batches for training
batch_size = 32
n_train = X_train.shape[0]


def ceil(a, b):
    return -(-a//b)


def next_batch(batch_num):
    batch = X_train[batch_num*batch_size: (batch_num+1)*batch_size, :]
    return batch


# Training Parameters
eta = 0.001
num_epochs = 200

# Network Parameters
num_hidden_1 =  380  # nodes in 1st Hidden Layer
num_linear = 80       # nodes in linear layer (reduced Representation)
num_input = X_train.shape[1]

# placeholder for input data
X = tf.placeholder("float", [None, num_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1], seed=10)*.01),
    'encoder_lin_w': tf.Variable(tf.truncated_normal([num_hidden_1, num_linear], seed=20)*.01),
    'decoder_h1': tf.Variable(tf.truncated_normal([num_linear, num_hidden_1], seed=30)*.01),
    'decoder_o': tf.Variable(tf.random_normal([num_hidden_1, num_input], seed=40)*.01),

}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1], seed=50)*.01),
    'linear_b': tf.Variable(tf.random.normal([num_linear], seed=60)*.01),
    'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1], seed=70)*.01),
    'decoder_o': tf.Variable(tf.random_normal([num_input], seed=80)*.01)
}


# Encoder
def encoder(x):
    # use Sigmoid Activation for Hidden layer 1 and Hidden Layer 2 of the Encoder
    layer_1 = tf.nn.softplus(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    linear_layer = tf.add(tf.matmul(layer_1, weights['encoder_lin_w']), biases['linear_b'])
    #linear_layer = tf.add(tf.matmul(x, weights['encoder_lin_w']), biases['linear_b'])
    return linear_layer


# Decoder
def decoder(x):
    # use Sigmoid Activation for Hidden layer 1 and Hidden Layer 2 of the Decoder
    layer_1 = tf.nn.softplus(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    output = tf.add(tf.matmul(layer_1, weights['decoder_o']), biases['decoder_o'])
    #output = tf.add(tf.matmul(x, weights['decoder_o']), biases['decoder_o'])
    return output


# construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Predicted output
y_pred = decoder_op
# Target output
y_true = X

# set loss and minimizer
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(eta).minimize(loss)

# Initialize the variables
init = tf.global_variables_initializer()

# Start Training
sess = tf.Session()
sess.run(init)

num_batches_train = ceil(n_train, batch_size)
for i in range(1, num_epochs+1):
    curr_epoch_loss = 0
    for j in range(0, num_batches_train):
        batch_x = next_batch(j)
        _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})
        curr_epoch_loss += l

    current_epoch_loss = sess.run(loss, feed_dict={X: X_train})
    current_epoch_loss_val = sess.run(loss, feed_dict={X: X_remain})
    print(" Epochs: %d. Loss = %g, val_loss: %g" % (i, current_epoch_loss, current_epoch_loss_val))


# Evaluate the error on Train data
val_recons = sess.run(decoder_op, feed_dict={X: X_train})
y_true = X_train
y_pred = val_recons
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
print('Train loss', sess.run(loss))




#
# # Evaluate the error on Test data
# val_recons = sess.run(decoder_op, feed_dict={X: X_test})
# y_true = X_test
# y_pred = val_recons
# loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
# print(sess.run(loss))

# with open('./features_supervised.csv', 'r') as csvFile:
#     X_remain = list(csv.reader(csvFile))
# X_remain = np.array(X_remain, dtype=float)
n_remain = X_remain.shape[0]

# write reduced dimension data in a file
X_red = sess.run(decoder_op, feed_dict={X: X_train})
with open('X_red_train.txt', 'w') as f:
    for i in range(n_train):
        for j in range(num_linear):
            f.write("%s " % X_red[i][j])
        f.write("\n")

X_red = sess.run(decoder_op, feed_dict={X: X_remain})
with open('X_red_remain.txt', 'w') as f:
    for i in range(n_remain):
        for j in range(num_linear):
            f.write("%s " % X_red[i][j])
        f.write("\n")
