import os
import numpy as np
import csv

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
# image_features = np.array(image_features)
# labels = np.array(labels)
# plt.hist(labels, 10)
# plt.show()
# print(image_features.shape)
# print(labels.shape)

with open('./features_unsupervised.csv', 'r') as csvFile:
    x_train = list(csv.reader(csvFile))
x_train = np.array(x_train, dtype=float)
n_train = x_train.shape[0]


#### Perform PCA on the Data
mean_X = np.sum(x_train, axis=0)/n_train
X_mean_centered = x_train - mean_X
cov_X = np.matmul(X_mean_centered.T, X_mean_centered)/(n_train - 1.)
#print(np.cov(image_features.T) == cov_X)
values, vectors = np.linalg.eig(cov_X)
idx = values.argsort()[::-1]
values = values[idx]
vectors = vectors[:, idx]
var_total = np.sum(values)
# var_target = .95 * var_total
#
# v = 0
# npc = 0     # number of Principal Components
#
# while v < var_target:
#     v = v + values[npc]
#     npc += 1
npc = 260
print("Number of PC ", npc)

with open('./features_supervised.csv', 'r') as csvFile:
    x_remain = list(csv.reader(csvFile))
x_remain = np.array(x_remain, dtype=float)
n_remain = x_remain.shape[0]

# Reduce the dimension of the train data by projecting data on Eigen Vectors #####
X_red_train = np.matmul(vectors[:, 0:npc].T, X_mean_centered.T)
X_red = X_red_train.T.real


with open('X_red_train.txt', 'w') as f:
    for i in range(n_train):
        for j in range(npc):
            f.write("%s " % X_red[i][j])
        f.write("\n")

# Reduce the dimension of the test data by projecting data on Eigen Vectors #####
X_mean_centered_remain = x_remain - mean_X
X_red_remain = np.matmul(vectors[:, 0:npc].T, X_mean_centered_remain.T)
X_red = X_red_remain.T.real

with open('X_red_remain.txt', 'w') as f:
    for i in range(n_remain):
        for j in range(npc):
            f.write("%s " % X_red[i][j])
        f.write("\n")