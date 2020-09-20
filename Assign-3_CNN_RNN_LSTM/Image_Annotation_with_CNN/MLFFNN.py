from __future__ import print_function, division
import os
import numpy as np
import tensorflow as tf


dirname = os.path.dirname(__file__)
X_train_file = dirname + '/train_x_VGG.txt'
X_val_file = dirname + '/val_x_VGG.txt'
X_test_file = dirname + '/test_x_VGG.txt'
Y_train_file = dirname + '/train_Y_VGG.txt'
Y_val_file = dirname + '/val_Y_VGG.txt'
Y_test_file = dirname + '/test_Y_VGG.txt'

X_train = np.array(np.loadtxt(X_train_file), dtype=float)
X_val = np.array(np.loadtxt(X_val_file), dtype=float)
X_test = np.array(np.loadtxt(X_test_file), dtype=float)
Y_train = np.array(np.loadtxt(Y_train_file), dtype=float)
Y_val = np.array(np.loadtxt(Y_val_file), dtype=float)
Y_test = np.array(np.loadtxt(Y_test_file), dtype=float)


dim = X_train.shape[1]

n_train = X_train.shape[0]
n_val = X_val.shape[0]
n_test = X_test.shape[0]

threshold = .01

def ceil(a, b):
    return -(-a//b)


def next_batch(batch_num):
    batch_x = X_train[batch_num*batch_size: (batch_num+1)*batch_size, :]
    batch_y = Y_train[batch_num*batch_size: (batch_num+1)*batch_size, :]
    return batch_x, batch_y


def getPRF(y_true, result):

    '''Micro : Datapointwise
       Macro : Classwise /labelwise
    '''
    f1s = [0, 0]
    pre = [0, 0]
    rec = [0, 0]
    y_true = tf.cast(y_true, tf.float64)
    result = tf.cast(result, tf.float64)

    list = [0, 1]
    for i, axis in enumerate([list, 0]):
        TP = tf.count_nonzero(result * y_true, axis=axis)
        FP = tf.count_nonzero(result * (y_true - 1), axis=axis)
        FN = tf.count_nonzero((result - 1) * y_true, axis=axis)

        precision = TP / (TP + FP)
        precision_avg = tf.reduce_mean(precision)
        recall = TP / (TP + FN)
        recall_avg = tf.reduce_mean(recall)
        f1 = 2 * precision * recall / (precision + recall)

        f1s[i] = tf.reduce_mean(f1)
        pre[i] = precision_avg
        rec[i] = recall_avg

    macro, micro = f1s
    return pre[1], rec[1], micro, pre[0], rec[0], macro

# Parameters
learning_rate = 0.0001
batch_size = 16


# Network Parameters
n_hidden_1 = 1000  # 1st layer number of neurons
n_hidden_2 = 500    #t 2nd layer number of neurons
num_input = dim  #  number of nodes in input layer
num_classes = 6    #  total classes (0-4 )

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
    layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    # apply DropOut to hidden layer 1
    # drop_out_1 = tf.nn.dropout(layer_1, keep_prob)
    # Hidden fully connected layer with 256 neurons
    # z_BN2 = tf.matmul(drop_out_1, weights['h2'])
    # batch_mean2, batch_var2 = tf.nn.moments(z_BN2, [0])
    # scale2 = tf.Variable(tf.ones([n_hidden_2]))
    # beta2 = tf.Variable(tf.zeros([n_hidden_2]))
    # BN2 = tf.nn.batch_normalization(z_BN2, batch_mean2, batch_var2, beta2, scale2, epsilon)
    # layer_2 = tf.nn.leaky_relu(BN2)
    layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
    # apply DropOut to hidden layer 2
    # drop_out_2 = tf.nn.dropout(layer_2, keep_prob)
    # Output fully connected layer with a neuron for each class
    out_layer = tf.nn.sigmoid(tf.matmul(layer_2, weights['out']) + biases['out'])
    # out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


# Construct model
# logits = neural_net(X, keep_prob)
#
# # Define loss and optimizer
# loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))

# L2 loss
#reg = tf.nn.l2_loss(weights['h1']) + tf.nn.l2_loss(weights['h2'])
#loss = tf.reduce_mean(loss_op + reg * beta)
#loss = tf.reduce_mean(loss_op)
y_pred = neural_net(X, keep_prob)
y_true = Y
loss = tf.reduce_mean(tf.reduce_sum(tf.pow(y_true - y_pred, 2), axis=1))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

## Evaluate the model
result = tf.cast(y_pred + 0.5, tf.int32)

# Evaluate model ( for dropout to be disabled)
correct_pred = tf.equal(result, tf.cast(y_true, tf.int32))
accuracy = tf.reduce_mean(tf.reduce_sum(tf.cast(correct_pred, tf.float64), axis=1))
pre_mic, pre_mac, rec_mic, rec_mac, micro, macro = getPRF(y_true, result)

num_batches_train = ceil(n_train, batch_size)
num_epochs = 10
# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    # current_epoch_loss = 500.0
    # previous_epoch_loss = 1000.0
    # current_epoch_loss_val = 500.0
    # previous_epoch_loss_val = 1000.0
    # epochs = 0

    # while current_epoch_loss > 1.1:
    for e in range(num_epochs):
        # previous_epoch_loss = current_epoch_loss
        # current_epoch_loss = 0
        # current_epoch_loss_val = previous_epoch_loss_val
        # current_epoch_loss_val  = 0

        for j in range(0, num_batches_train):
            batch_x, batch_y = next_batch(j)
            summary, train_loss = sess.run([train_op, loss], feed_dict={X: batch_x, Y: batch_y})

        current_epoch_loss = sess.run(loss, feed_dict={X: X_train, Y: Y_train})
        current_epoch_loss_val = sess.run(loss, feed_dict={X: X_val, Y: Y_val})

        print(" Epochs: %d. Loss = %g, val_loss: %g" % (e, current_epoch_loss, current_epoch_loss_val))
        # epochs = epochs + 1

    print("Optimization Finished!")

    # Calculate Precision Recall and F1-score for  train images
    pred_mic, reca_mic, micro_f1, pred_mac, reca_mac, macro_f1 = sess.run([pre_mic, rec_mic, micro, pre_mac, rec_mac, macro], feed_dict={X: X_train, Y: Y_train})
    print("Train:\n Micro: precision : %g, recall: %g, f-score: %g, \n \
            Macro: precision : %g, recall: %g,\n f-score: %g, " % (pred_mic, reca_mic, micro_f1, pred_mac, reca_mac, macro_f1))

    # Calculate Precision Recall and F1-score for  val images
    pred_mic, reca_mic, micro_f1, pred_mac, reca_mac, macro_f1 = sess.run([pre_mic, rec_mic, micro, pre_mac, rec_mac, macro], feed_dict={X: X_val, Y: Y_val})
    print("Val :\n Micro: precision : %g, recall: %g, f-score: %g, \n \
           Macro: precision : %g, recall: %g,\n f-score: %g, " % (pred_mic, reca_mic, micro_f1, pred_mac, reca_mac, macro_f1))

    # # Calculate Precision Recall and F1-score for  test images
    pred_mic, reca_mic, micro_f1, pred_mac, reca_mac, macro_f1 = sess.run([pre_mic, rec_mic, micro, pre_mac, rec_mac, macro], feed_dict={X: X_test, Y: Y_test})
    print("Test:\n Micro: precision : %g, recall: %g, f-score: %g, \n \
           Macro: precision : %g, recall: %g,\n f-score: %g, " % ( pred_mic, reca_mic, micro_f1, pred_mac, reca_mac, macro_f1))

    train_accuracy = sess.run([accuracy], feed_dict={X: X_train, Y: Y_train})
    print("Train Accuracy: %g " % (train_accuracy[0]))

    test_accuracy = sess.run([accuracy], feed_dict={X: X_test, Y: Y_test})
    print("Test Accuracy: %g " % (test_accuracy[0]))

    '''Writing predictions to file'''
    y = sess.run([result], feed_dict={X: X_train, Y: Y_train})
    with open(dirname + '/train_predicted.txt', 'w') as f:
        for item in np.array(y):
            for value in item:
                f.write("%s " % value)
            f.write("\n")

    y = sess.run([result], feed_dict={X: X_test, Y: Y_test})
    with open(dirname + '/test_predicted.txt', 'w') as f:
        for item in np.array(y):
            for value in item:
                f.write("%s " % value)
            f.write("\n")

