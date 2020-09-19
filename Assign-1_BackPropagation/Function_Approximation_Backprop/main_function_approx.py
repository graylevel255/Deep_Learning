import Backprop as bp
import os
import numpy as np
import matplotlib.pyplot as plt


def normalize(x, x_min, x_max):
    # perform min max normalization on input data
    return (x-x_min)/(x_max.astype(float) - x_min.astype(float))

# Read the input data
dirname = os.path.dirname(__file__)
train = dirname + '/train.txt'
train = np.loadtxt(train)
ip_train = train[:, 0:8]
op_train = train[:, 8:]
n_train = len(op_train)

test = dirname + '/test.txt'
test = np.loadtxt(test)
ip_test = test[:, 0:8]
op_test = test[:, 8:]
n_test = len(op_test)

# Normalizing training data

x_max = np.max(ip_train, axis=0)
x_min = np.min(ip_train, axis=0)
y_max = np.max(op_train, axis=0)
y_min = np.min(op_train, axis=0)

ip_train = normalize(ip_train, x_min, x_max)
ip_test = normalize(ip_test, x_min, x_max)

#ip_train = ip_list[:n_train]
#op_train = op_list[:n_train]

#ip_test = ip_list[n_train:]
#op_test = op_list[n_train:]

## write split data into a train and test file

nn = bp.NeuralNet([8, 6, 2])
eta = 1
threshold = [1e-4]
old_error = [1000, 1000]
epochs = 0
ep = []
with open('mse_error.txt', 'w') as f:

    while True:
        epochs += 1
        new_error = nn.update_weights_stochastic(ip_train, op_train, n_train, eta)
        del_error = abs(new_error - old_error)
        print epochs
        ep.append(epochs)
        print new_error

        # write error to file
        f.write("%s\n" % new_error)
        if np.any(del_error < threshold) or epochs >= 2000:
            break
        old_error = new_error
print(epochs)
print(nn.get_weights())
print(nn.get_bias())

# Calculate Mean Squared Error on train data
y, mse = nn.test(ip_train, op_train)
ax1 = plt.subplot(221)
ax1.scatter(np.array(op_train)[:, 0], np.array(y)[:, 0], s=15)
ax1.set_xlim([0, 50])
ax1.set_ylim([0, 50])
ax1.plot(ax1.get_xlim(), ax1.get_xlim(), c='r')

ax2 = plt.subplot(222)
ax2.scatter(np.array(op_train)[:, 1], np.array(y)[:, 1], s=15)
ax2.set_xlim([0, 50])
ax2.set_ylim([0, 50])
ax2.plot(ax2.get_xlim(), ax2.get_xlim(),c='r')

ax1.set_xlabel('Target Output')
ax1.set_ylabel('Model Output')
ax2.set_xlabel('Target Output')
ax2.set_ylabel('Model Output')
ax1.set_title('Training Y1')
ax2.set_title('Training Y2')


print(mse)

# Calculate Mean Squared Error on test data
y, mse = nn.test(ip_test, op_test)
ax3 = plt.subplot(223)
#plt.rcParams["legend.markerscale"] = 0.3
ax3.scatter(np.array(op_test)[:, 0], np.array(y)[:, 0], s=15)
ax3.set_xlim([0, 50])
ax3.set_ylim([0, 50])
ax3.plot(ax3.get_xlim(), ax3.get_xlim(), c='r')

ax4 = plt.subplot(224)
#plt.rcParams["legend.markerscale"] = 2

ax4.scatter(np.array(op_test)[:, 1], np.array(y)[:, 1], s=15)
ax4.set_xlim([0, 50])
ax4.set_ylim([0, 50])
ax3.set_xlabel('Target Output')
ax3.set_ylabel('Model Output')
ax4.set_xlabel('Target Output')
ax4.set_ylabel('Model Output')
ax3.set_title('Test Y1')
ax4.set_title('Test Y2')
ax4.plot(ax4.get_xlim(), ax4.get_xlim(), c='r')

plt.subplots_adjust(left=0.25, bottom=0.1, right=0.75, top=0.925, wspace=0.35, hspace=0.35)
plt.show()
print epochs
print(mse)
#print ip_train

# Plot error vs epochs

err = dirname + '/mse_error.txt'
mse_error = np.loadtxt(err)
#print type(ep)
#print type(mse_error)

epo = np.array(ep)
#print type(epo)
plt.plot(epo, mse_error, color='r')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Loss v/s Epochs for training'], loc='upper right')
plt.title('Loss v/s Epochs using for Training Data')
plt.show()
