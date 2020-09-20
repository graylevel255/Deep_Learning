from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# read the input data


def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    image_features = []
    label = 0
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f)
                      for f in os.listdir(label_directory) if f.endswith(".txt")]

        for f in file_names:
            feature = np.loadtxt(f, delimiter=' ')
            image_features.append(feature.flatten())
            labels.append(label)
        label += 1
    return image_features, labels


ROOT_PATH = './SingleLabelImageFeatures/Features/'

image_features, labels = load_data(ROOT_PATH)
image_features = np.array(image_features)
labels = np.array(labels)
# plt.hist(labels, 10)
# plt.show()
print(image_features.shape)
print(labels.shape)
n = image_features.shape[0]
# Splitting the data and Train, Val and Test
initial_split = .3
X_train, X_remain, y_train, y_remain = train_test_split(image_features, labels, test_size=initial_split, random_state=2)
new_split = 10.0/(initial_split*100.)
X_val, X_test, y_val, y_test = train_test_split(X_remain, y_remain, test_size=(1-new_split), random_state=2)

# Making mini batches for training
batch_size = 32
n_train = X_train.shape[0]


def ceil(a, b):
    return -(-a//b)


def next_batch(batch_num):
    batch = X_train[batch_num*batch_size: (batch_num+1)*batch_size, :]
    return batch


# Training Parameters
eta = 0.01
num_epochs = 400
display_step = 100

# Network Parameters
num_hidden_1 = 512     # nodes in 1st Hidden Layer
num_hidden_2 = 256      # nodes in 2nd Hidden Layer
num_hidden_3 = 200      # nodes in 2nd Hidden Layer
num_linear = 120        # nodes in linear layer (reduced Representation)
num_input = X_train.shape[1]

# placeholder for input data
X = tf.placeholder("float", [None, num_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1], seed=1)*.01),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2], seed=2)*.01),
    'encoder_h3': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_3], seed=3)*.01),
    'encoder_lin_w': tf.Variable(tf.random_normal([num_hidden_3, num_linear], seed=4)*.01),
    'decoder_h1': tf.Variable(tf.random_normal([num_linear, num_hidden_3], seed=5)*.01),
    'decoder_h2': tf.Variable(tf.random_normal([num_hidden_3, num_hidden_2], seed=6)*.01),
    'decoder_h3': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1], seed=7)*.01),
    'decoder_o': tf.Variable(tf.random_normal([num_hidden_1, num_input], seed=8)*.01),

}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1], seed=9)*.01),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2], seed=10)*.01),
    'encoder_b3': tf.Variable(tf.random_normal([num_hidden_3], seed=11)*.01),
    'linear_b': tf.Variable(tf.random_normal([num_linear], seed=16)*.01),
    'decoder_b1': tf.Variable(tf.random_normal([num_hidden_3], seed=12)*.01),
    'decoder_b2': tf.Variable(tf.random_normal([num_hidden_2], seed=13)*.01),
    'decoder_b3': tf.Variable(tf.random_normal([num_hidden_1], seed=14)*.01),
    'decoder_o': tf.Variable(tf.random_normal([num_input], seed=15)*.01)

}


# Encoder
def encoder(x):
    # use Sigmoid Activation for Hidden layer 1 and Hidden Layer 2 of the Encoder
    layer_1 = tf.nn.softplus(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    layer_2 = tf.nn.softplus(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    layer_3 = tf.nn.softplus(tf.add(tf.matmul(layer_2, weights['encoder_h3']), biases['encoder_b3']))
    linear_layer = tf.add(tf.matmul(layer_3, weights['encoder_lin_w']), biases['linear_b'])

    return linear_layer


# Decoder
def decoder(x):
    # use Sigmoid Activation for Hidden layer 1 and Hidden Layer 2 of the Decoder
    layer_1 = tf.nn.softplus(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    layer_2 = tf.nn.softplus(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
    layer_3 = tf.nn.softplus(tf.add(tf.matmul(layer_2, weights['decoder_h3']), biases['decoder_b3']))
    output = tf.add(tf.matmul(layer_3, weights['decoder_o']), biases['decoder_o'])

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
    print("EPOCH: ", i)
    for j in range(0, num_batches_train):
        batch_x = next_batch(j % num_batches_train)
        _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})

        if j%display_step == 0 or j == 1:
            print('EPOCH%i Step%i: Minibatch Loss: %f' % (i, j, l))


# Evaluate the error on Train data
val_recons = sess.run(decoder_op, feed_dict={X: X_train})
y_true = X_train
y_pred = val_recons
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
print(sess.run(loss))


# Evaluate the error on validation data
val_recons = sess.run(decoder_op, feed_dict={X: X_val})
y_true = X_val
y_pred = val_recons
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
print(sess.run(loss))

# Evaluate the error on Test data
val_recons = sess.run(decoder_op, feed_dict={X: X_test})
y_true = X_test
y_pred = val_recons
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
print(sess.run(loss))

# write reduced dimension data in a file X_redhl.txt
X_red = sess.run(decoder_op, feed_dict={X: image_features})

with open('X_redhl.txt', 'w') as f:
    for i in range(n):
        for j in range(num_linear):
            f.write("%s " % X_red[i][j])
        f.write("%s\n" % labels[i])