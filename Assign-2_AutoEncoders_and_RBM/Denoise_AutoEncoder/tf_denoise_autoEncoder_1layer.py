import os
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

my_path = os.path.dirname(os.path.realpath(__file__))  #gets current path of the file
noisy_data = my_path + "/noisy_images_15.csv"
clean_data = my_path + "/clean_images.csv"   #gets path of the file
clean_images = np.genfromtxt(clean_data, delimiter=',')
noisy_images = np.genfromtxt(noisy_data, delimiter=',')

print(type(clean_images), type(noisy_images))
print(clean_images.shape, noisy_images.shape)

#print("DONE")


## Splitting the data and Train, Test and Val
X_train, X_test, y_train, y_test = train_test_split(noisy_images, clean_images, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.33, random_state=42)

# Read split data from files
#
# with open('./train_data.csv', 'r') as csvfile:
#     train = list(csv.reader(csvfile))
#     X_train = np.array(train)
# with open('./test_data.csv', 'r') as csvfile:
#     test = list(csv.reader(csvfile))
#     X_test = np.array(test)
# with open('./val_data.csv', 'r') as csvfile:
#     val = list(csv.reader(csvfile))
#     X_val = np.array(val)
#
# # READ LABELS
# with open('./train_labels.csv', 'r') as csvfile:
#     train_labels = list(csv.reader(csvfile))
#     y_train = np.array(train_labels)
# with open('./test_labels.csv', 'r') as csvfile:
#     test_labels = list(csv.reader(csvfile))
#     y_test = np.array(test_labels)
# with open('./val_labels.csv', 'r') as csvfile:
#     val_labels = list(csv.reader(csvfile))
#     y_val = np.array(val_labels)

# Making mini batches for training

n_train = X_train.shape[0]
batch_size = 100

# Training Parameters

learning_rate = 0.0001
epochs = 1000
# display_epoch = 500

# Network skeleton

ip_dim = X_train.shape[1]
h1 = 350
# h2 = 128
# h3 = 100

ip_data = tf.placeholder("float", [None, ip_dim])
op_data = tf.placeholder("float", [None, ip_dim])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([ip_dim, h1], seed = 1)*0.01),
    'decoder_h1': tf.Variable(tf.random_normal([h1, ip_dim], seed = 4)*0.01),

}

biases = {
    'encoder_b1': tf.Variable(tf.random_normal([h1], seed = 7)*0.01),
    'decoder_b1': tf.Variable(tf.random_normal([ip_dim], seed = 11) * 0.01),
}

# Building the encoder and decoder

def encoder(x):

    enc_layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))

    return enc_layer1

def decoder(x):

    dec_layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))

    return dec_layer1

# For batch mode

def ceil(a,b):
    return -(-a//b)

def next_batch(batch_num):
    batch = X_train[batch_num*batch_size: (batch_num+1)*batch_size, :]
    return batch

# to pick true images in a batch
def next_batch_ytrue(batch_num):
    batch = y_train[batch_num*batch_size: (batch_num+1)*batch_size, :]
    return batch

# For val data
def next_batch_val(batch_num):
    batch = X_val[batch_num*batch_size: (batch_num+1)*batch_size, :]
    return batch

# to pick true images in a batch
def next_batch_ytrue_val(batch_num):
    batch = y_val[batch_num*batch_size: (batch_num+1)*batch_size, :]
    return batch


# For test data
def next_batch_test(batch_num):
    batch = X_test[batch_num*batch_size: (batch_num+1)*batch_size, :]
    return batch

# to pick true images in a batch
def next_batch_ytrue_test(batch_num):
    batch = y_test[batch_num*batch_size: (batch_num+1)*batch_size, :]
    return batch



# Build the model

encoder_op = encoder(ip_data)
decoder_op = decoder(encoder_op)

print("ENCODER SHAPE : ", encoder_op.shape)
#encoder_ytrue_op = encoder(op_data)
#decoder_ytrue_op = decoder(encoder_ytrue_op)

# Predict outputs

y_pred = decoder_op
y_true = op_data

# Loss and optimization functions

loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Initialize the variables

init = tf.global_variables_initializer()

# Start training by creating a new tf session
ep = 1


with tf.Session() as sess:
    sess.run(init)
    num_batches_train = ceil(n_train, batch_size)
    for i in range(1, epochs + 1):
        curr_ep_loss = 0
        prev_ep_loss = 100
        # while prev_ep_loss - curr_ep_loss > 1e-3:
        #     prev_ep_loss = curr_ep_loss
        #     curr_ep_loss = 0
        for j in range(num_batches_train):
            batch_x = next_batch(j % num_batches_train)
            y_true = next_batch_ytrue(j % num_batches_train)
            _, l = sess.run([optimizer, loss], feed_dict={ip_data: batch_x, op_data: y_true})
            curr_ep_loss += l
        # if i % display_epoch == 0 or i == 1:
        print('Step%i: Epoch Loss: %f' % (i, curr_ep_loss/num_batches_train))

    # for i in range(1, epochs + 1):
    #     batch_x, _ = tf.train.next_batch(batch_size)
    #
    #     _, l = sess.run([optimizer, loss], feed_dict={ip_data: batch_x})
    #
        # Display logs per step
        # if i % display_epoch == 0 or i == 1:
        #     print('Step %i: Minibatch Loss: %f' % (i, l))

    # Encode and decode images from validation set and visualize their reconstruction.

    n = 4
    canvas_noisy = np.empty((28 * n, 28 * n))
    canvas_recon = np.empty((28 * n, 28 * n))
    canvas_original = np.empty((28 * n, 28 * n))
    for i in range(n):
        # MNIST test set
        batch_x = next_batch_test(i+1)
        batch_y = next_batch_ytrue_test(i+1)
        # Encode and decode the digit image

        g = sess.run(decoder_op, feed_dict={ip_data: batch_x})
        g = g.round()

        # # Display original images
        for j in range(n):
            # Draw the original digits
            canvas_original[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                batch_y[j].reshape([28, 28])

        # # Display noisy images
        for j in range(n):
            # Draw the original digits
            canvas_noisy[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                batch_x[j].reshape([28, 28])

        # Display reconstructed images
        for j in range(n):
            # Draw the reconstructed digits
            canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                g[j].reshape([28, 28])

    print("Original Validation Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_original, origin="upper", cmap="gray")
    plt.show()

    print("Noisy Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_noisy, origin="upper", cmap="gray")
    plt.show()

    print("Reconstructed Validation Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_recon, origin="upper", cmap="gray")
    plt.show()

    # Print validation Loss

    val_recon = sess.run(decoder_op, feed_dict={ip_data: X_train})
    val_true = X_train
    val_pred = val_recon
    val_loss = tf.reduce_mean(tf.pow(val_true - val_pred, 2))
    print("Validation Loss :")
    print(sess.run(val_loss))

    # Print test loss
    test_recon = sess.run(decoder_op, feed_dict={ip_data: X_test})
    test_true = X_test
    test_pred = test_recon
    test_loss = tf.reduce_mean(tf.pow(test_true - test_pred, 2))
    print("Test Loss :")
    print(sess.run(test_loss))