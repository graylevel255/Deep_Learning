import numpy as np
from copy import copy, deepcopy

np.seterr(divide='ignore', invalid='ignore')
#import bigfloat
#bigfloat.exp(50000,bigfloat.precision(1000))
# -> BigFloat.exact('2.9676283840236670689662968052896e+2171', precision=100)

class Sigmoid :

    @staticmethod
    def activ_fun(a, layer_num, beta=1, delta=1):
         if layer_num == 0:
             activ_val = beta*a
         else:
             activ_val = a
         #n = len(activ_val)
         #phi_val = np.zeros(n)
         phi_val = 1/(1 + np.exp(-activ_val))
         return phi_val

    @staticmethod
    def activ_fun_der(a, layer_num, beta=1, delta=1):

        all_ones = np.ones(len(a))
        temp = Sigmoid.activ_fun(a, layer_num, beta, delta)
        #temp = temp.reshape(temp.shape[0], 1)
        #all_ones = all_ones.reshape((all_ones.shape[0], 1))
        if layer_num == 0:
            deriv = beta*temp*(all_ones-temp)
        else:
            deriv = temp*(all_ones - temp)
        #return beta*phi_val*(1 - phi_val)
        return deriv

class TanH :

    @staticmethod
    def activ_fun(a, layer_num, beta=1, delta=1):
        if layer_num == 0:
            activ_val = beta * a
        else:
            activ_val = a;
        #e_val1 = np.exp(activ_val)
        #e_val2 = np.exp(-activ_val)
        #num = e_val1 - e_val2
        #denom = e_val1 + e_val2
        #phi_val = num/denom
        phi_val = np.tanh(activ_val)
        return phi_val

    @staticmethod
    def activ_fun_der(a, layer_num, beta=1, delta=1):
        all_ones = np.ones(len(a))
        temp = np.tanh(a)
        if layer_num == 0:
            deriv = beta*(all_ones + temp)*(all_ones - temp)
        else:
            deriv = (all_ones + temp)*(all_ones - temp)
        return deriv


class ReLU :

    @staticmethod
    def activ_fun(a, layer_num, beta=1, delta=1):
        #a[a < 0] = 0
        #a[a >= 0] = a
        #return a
        return np.maximum(0,a)

    @staticmethod
    def activ_fun_der(a, layer_num, beta=1, delta=1):
        temp = deepcopy(a)
        temp[temp <= 0] = 0
        temp[temp > 0] = 1
        return temp

class SoftPlus :

    @staticmethod
    def activ_fun(a, layer_num, beta=1, delta=1):
        phi_val = np.log(1 + np.exp(a))
        return phi_val

    @staticmethod
    def activ_fun_der(a, layer_num, beta=1, delta=1):
        phi_val_der = 1 / (1 + np.exp(-a))
        return phi_val_der

class ELU :

    @staticmethod
    def activ_fun(a, layer_num, beta=1, delta=1):
        #t = deepcopy(a)
        #t[t > 0] = t
        #t[t <= 0] = delta*(np.exp(t[t<=0]) - 1)
        return np.where(a < 0, delta * (np.exp(a) - 1), a)
        #return t

    @staticmethod
    def activ_fun_der(a, layer_num, beta=1, delta=1):
        td = deepcopy(a)
        td[td > 0] = 1
        td[td <= 0] = delta*np.exp(td[td <= 0])
        return td


class Softmax :

    @staticmethod
    def activ_fun(a, layer_num, beta=1, delta=1):
        return np.exp(a) / np.sum(np.exp(a), axis=0)

    @staticmethod
    def activ_fun_der(a, layer_num, beta=1, delta=1):
            a = np.exp(a)
            s = sum(a)
            d = np.zeros(len(a))
            for x in range(len(a)):
                d[x] = (s - a[x]) / s ** 2
            return d

class SSE:
    @staticmethod
    def activ_fun(a, layer_num, beta=1, delta=1):
        return a

    @staticmethod
    def activ_fun_der(a, layer_num, beta=1, delta=1):
        return 1

    @staticmethod
    def delta(y_predicted, y_desired, activation_deriv):
        return np.array(y_predicted - y_desired) * np.array(activation_deriv)

