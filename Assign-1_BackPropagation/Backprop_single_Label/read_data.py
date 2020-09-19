import os
import numpy as np
from sklearn.model_selection import train_test_split

from random import shuffle

dirname = os.path.dirname(__file__)
#print dirname

input = dirname + '/data.txt'
label = dirname + '/label.txt'
ip_file = np.loadtxt(input)
lb_file = np.loadtxt(label, delimiter=',')
length = len(ip_file)
oneOfK = [np.zeros(5) for i in range(length)]
for i in range(length):
    oneOfK[i][int(lb_file[i])] = 1
total_data = np.column_stack((ip_file, oneOfK))
total_data = np.column_stack((total_data, lb_file))
# np.random.shuffle(total_data)

with open('total_data.txt', 'w') as f:
    for item in np.array(total_data):
        for value in item:
            f.write("%s " % value)
        f.write("\n")
#
