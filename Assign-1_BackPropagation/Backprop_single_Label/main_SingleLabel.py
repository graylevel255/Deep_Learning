import Backprop_SingleLabel as bp
import os
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


def normalize(x, x_min, x_max):
    # perform min max normalization on input data
    return (x-x_min)/(x_max.astype(float) - x_min.astype(float))


# Read the input data
dirname = os.path.dirname(__file__)
train_data = dirname + '/train.txt'
train = np.loadtxt(train_data)
ip_train = train[:, 0:60]
op_train = train[:, 60:65]
label_train = train[:, 65:]
n_train = len(op_train)

test_data = dirname + '/test.txt'
test = np.loadtxt(test_data)
ip_test = test[:, 0:60]
ip_test = ip_test
op_test = test[:, 60:65]
label_test = test[:, 65:]
n_test = len(op_test)


# # Normalizing training data
x_max = np.max(ip_train, axis=0)
x_min = np.min(ip_train, axis=0)
ip_train = normalize(ip_train, x_min, x_max)
ip_test = normalize(ip_test, x_min, x_max)

# training the neural network
nn = bp.NeuralNet([60, 29, 5])
eta = .009
threshold = 0
epochs = 0.0
error = 0.1
error_old = 1000.0
error_test = 100

#while epochs < 100:
while True:
    #act, garbage = nn.update_weights_stochastic(ip_train, op_train, label_train,  n_train, eta, "adam")
    nn.update_weights_batch(ip_train, op_train, label_train, n_train, eta, "adam", alpha=.9)
    if epochs % 5 == 0.0:
        garg1, error = nn.test(ip_train, op_train)
        print "train", error
        garg, error1 = nn.test(ip_test, op_test)
        print "test", error1
        if np.abs(error_old - error) < 0.00001 or error1 - error_test > 0 :
            break
        else:
            error_old = error
            error_test = error1
    epochs += 1.0

print(epochs)
# print (nn.get_weights())
# print(nn.get_bias())
# Testing for training error

y_predict, sh1 = nn.test(ip_train, op_train)
y_predict = np.argmax(y_predict, axis=1)
cm = confusion_matrix(label_train, y_predict)
ac = accuracy_score(label_train, y_predict)
print(cm)
print(ac)

# Testing for test error
y_predict, sh2 = nn.test(ip_test, op_test)
y_predict = np.argmax(y_predict, axis=1)
cm = confusion_matrix(label_test, y_predict)
ac = accuracy_score(label_test, y_predict)
print(cm)
print(ac)
# print(y_predict)
