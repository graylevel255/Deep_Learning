import os
import numpy as np

dirname = os.path.dirname(__file__)
#print dirname

input = dirname + '/energy-data.txt'
ip_file = np.loadtxt(input)

#print ip_file.shape

input = ip_file[:,0:8]
output = ip_file[:,8:]
#print input.shape
#print output.shape

#print type(input)

ip_list = input.tolist()
op_list = output.tolist()
