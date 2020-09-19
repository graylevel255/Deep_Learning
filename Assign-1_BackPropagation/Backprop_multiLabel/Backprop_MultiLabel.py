
import math
import numpy as np
import activation_functions as af


class NeuralNet:
    def __init__(self, layer_nodes, loss=af.SSE, activation_func=af.ELU):
        self.layer_nodes = layer_nodes
        self.layer_count = len(layer_nodes)
        self.loss = loss
        self.activation_func = activation_func
        np.random.seed(17)
        self.bias = [.01 * np.random.randn(i) for i in layer_nodes[1:]]
        np.random.seed(7)
        pairs = zip(layer_nodes[:self.layer_count-1], layer_nodes[1:])
        self.weights = [np.random.randn(p, c) * np.sqrt(2.0/c) for c, p in pairs]

        # delta_w and delta_b for momentum update.
        # this variable is also used for holding squares of gradients for Ada grad
        pairs = zip(layer_nodes[:self.layer_count - 1], layer_nodes[1:])
        self.delta_weights = [np.zeros((c, r)) for r, c in pairs]
        self.delta_bias = [np.zeros(n) for n in layer_nodes[1:]]
        # used to store exponentially moving average of change in weights for AdaDelta method
        # Needed for Ada Delta U_w
        pairs = zip(layer_nodes[:self.layer_count - 1], layer_nodes[1:])
        self.U_weights = [np.zeros((c, r)) for r, c in pairs]
        self.U_bias = [np.zeros(n) for n in layer_nodes[1:]]

        ## used for Adam update rule
        pairs = zip(layer_nodes[:self.layer_count - 1], layer_nodes[1:])
        self.delta_weights_hat = [np.zeros((c, r)) for r, c in pairs]
        self.delta_bias_hat = [np.zeros(n) for n in layer_nodes[1:]]

        pairs = zip(layer_nodes[:self.layer_count - 1], layer_nodes[1:])
        self.U_weights_hat = [np.zeros((c, r)) for r, c in pairs]
        self.U_bias_hat = [np.zeros(n) for n in layer_nodes[1:]]
        self.epoch_num = 0


    def __get_activation__(self, present_layer, current_activations):
        get_activation = np.dot(self.weights[present_layer], current_activations.transpose()) + self.bias[present_layer]
        return get_activation

    def __forward_pass__(self, incoming_layer):
        beta = 0.2
        delta = 4
        for i in range(0, self.layer_count-2):
            incoming_layer = self.activation_func.activ_fun(self.__get_activation__(i, incoming_layer), i, beta, delta)
        # output of the final/output layer using logistic function for MultiLabel
        incoming_layer = af.Sigmoid.activ_fun(self.__get_activation__(self.layer_count-2, incoming_layer), self.layer_count-2, beta, delta)
        return incoming_layer

    # x is the input to the neural network and y is the expected output
    def __back_propagation__(self, x, y):
        temp_activation = x
        temp2_matrix = [x]                                         # stores a i.e. activation value input to a layer
        activation_values = [x]                                    # stores phi(a) for every layer
        activation_derivatives = [np.zeros(self.layer_nodes[0])]   # stores derivatives for phi(a) w.r.t. a

        beta = 0.2
        delta = 4

        for j in range(0, self.layer_count-2):
            temp_activation = self.__get_activation__(j, temp_activation)
            temp2_matrix.append(temp_activation)
            activation_values.append(self.activation_func.activ_fun(temp_activation, j, beta, delta))
            activation_derivatives.append(self.activation_func.activ_fun_der(temp_activation, j, beta, delta))
            temp_activation = self.activation_func.activ_fun(temp_activation, j, beta, delta)
        # updating activation values and activation derivative for the last layer nodes
        temp_activation = self.__get_activation__(self.layer_count-2, temp_activation)
        temp2_matrix.append(temp_activation)
        activation_values.append(af.Sigmoid.activ_fun(temp_activation, self.layer_count-2, beta, delta))
        activation_derivatives.append(af.Sigmoid.activ_fun_der(temp_activation, self.layer_count-2, beta, delta))

        bias_gradient = [np.zeros(k.shape) for k in self.bias]  # local gradients for bias update
        weight_gradient = [np.zeros(a.shape) for a in self.weights]  # local gradients for weight update
        error = self.loss.delta(activation_values[self.layer_count - 1], y, activation_derivatives[self.layer_count - 1])  # (estimated-expected)(For sum of squared as of now)
        bias_gradient[self.layer_count - 2] = error
        weight_gradient[self.layer_count - 2] = np.dot(np.array([error]).transpose(),
                                                       np.array([activation_values[self.layer_count - 2]]))
        for l in reversed(range(1, self.layer_count - 1)):
            error = np.dot(self.weights[l].transpose(), error) * activation_derivatives[l]
            bias_gradient[l - 1] = error
            weight_gradient[l - 1] = np.dot(np.array([error]).transpose(), np.array([activation_values[l - 1]]))
            return bias_gradient, weight_gradient

    def update_weights_batch(self, x, y, N, eta, update_rule="delta", alpha=0.9, epsilon=1e-8):
        # N : size of training data
        # x : Input Data (features)
        # y : desired output

        # initialize weight gradients and bias gradients
        grad_weights = [np.zeros(r.shape) for r in self.weights]
        grad_bias = [np.zeros(n.shape) for n in self.bias]
        self.epoch_num += 1
        # calculating gradients by seeing each example and accumulating them
        for i in range(N):
            curr_grad_bias, curr_grad_weights = self.__back_propagation__(x[i], y[i])
            grad_weights = [g_old + g_new for g_old, g_new in zip(grad_weights, curr_grad_weights)]
            grad_bias = [g_old + g_new for g_old, g_new in zip(grad_bias, curr_grad_bias)]

        grad_weights = np.array(grad_weights) / float(N)
        grad_bias = np.array(grad_bias) / float(N)

        # update the weights and bias
        if update_rule == "delta":
            self.weights = [w_old - eta * grad_w for w_old, grad_w in zip(self.weights, grad_weights)]
            self.bias = [b_old - eta * grad_b for b_old, grad_b in zip(self.bias, grad_bias)]

        elif update_rule == "general_delta":
            self.weights = [w_old - eta * grad_w + alpha * del_w for w_old, grad_w, del_w in
                            zip(self.weights, grad_weights, self.delta_weights)]
            self.bias = [b_old - eta * grad_b + alpha * del_b for b_old, grad_b, del_b in
                         zip(self.bias, grad_bias, self.delta_bias)]
            self.delta_weights = [-eta * g_w + alpha * del_w for g_w, del_w in zip(grad_weights, self.delta_weights)]
            self.delta_bias = [-eta * g_w + alpha * del_w for g_w, del_w in zip(grad_bias, self.delta_bias)]

        elif update_rule == "ada_grad":
            # updating the parameters using the learning rate for each parameter  by adagrad
            #  updating r for next iteration, i.e. accumulating squares of gradients for next iteration
            self.delta_weights = [r2_w + (np.power(g_w, 2)) for r2_w, g_w in zip(self.delta_weights, grad_weights)]
            self.delta_bias = [r2_b + (np.power(g_b, 2)) for r2_b, g_b in zip(self.delta_bias, grad_bias)]

            self.weights = [
                w_old - (eta * np.array(grad_w, dtype=np.float)) / (np.sqrt(epsilon + np.array(r2_w, dtype=np.float)))
                for w_old, grad_w, r2_w in zip(self.weights, grad_weights, self.delta_weights)]
            self.bias = [
                b_old - (eta * np.array(grad_b, dtype=np.float)) / (np.sqrt(epsilon + np.array(r2_b, dtype=np.float)))
                for b_old, grad_b, r2_b in zip(self.bias, grad_bias, self.delta_bias)]

        elif update_rule == "rmsProp":
            rho = 0.95
            self.delta_weights = [rho * r2_w + (1 - rho) * (np.power(g_w, 2)) for r2_w, g_w in
                                  zip(self.delta_weights, grad_weights)]
            self.delta_bias = [rho * r2_b + (1 - rho) * (np.power(g_b, 2)) for r2_b, g_b in
                               zip(self.delta_bias, grad_bias)]

            self.weights = [
                w_old - ((eta * np.array(grad_w, dtype=np.float)) / (np.sqrt(epsilon + np.array(r2_w, dtype=np.float))))
                for w_old, grad_w, r2_w in zip(self.weights, grad_weights, self.delta_weights)]
            self.bias = [
                b_old - (eta * np.array(grad_b, dtype=np.float)) / (np.sqrt(epsilon + np.array(r2_b, dtype=np.float)))
                for b_old, grad_b, r2_b in zip(self.bias, grad_bias, self.delta_bias)]

        elif update_rule == "adaDelta":
            rho1 = 0.95
            rho2 = 0.9
            # rms values used in denominator same as Ada Grad
            self.delta_weights = [rho1 * r2_w + (1 - rho1) * (np.power(g_w, 2)) for r2_w, g_w in
                                  zip(self.delta_weights, grad_weights)]
            self.delta_bias = [rho1 * r2_b + (1 - rho1) * (np.power(g_b, 2)) for r2_b, g_b in
                               zip(self.delta_bias, grad_bias)]

            self.weights = [
                w_old - ((np.sqrt(np.array(U_w, dtype=np.float) + epsilon)) * np.array(grad_w, dtype=np.float)) / (
                    np.sqrt(epsilon + np.array(r2_w, dtype=np.float)))
                for w_old, U_w, grad_w, r2_w in zip(self.weights, self.U_weights, grad_weights, self.delta_weights)]
            self.bias = [
                b_old - ((np.sqrt(np.array(U_b, dtype=np.float) + epsilon)) * np.array(grad_b, dtype=np.float)) / (
                    np.sqrt(epsilon + np.array(r2_b, dtype=np.float)))
                for b_old, U_b, grad_b, r2_b in zip(self.bias, self.U_bias, grad_bias, self.delta_bias)]

            # Update U for the next iteration
            dW = [((np.sqrt(np.array(U_w, dtype=np.float) + epsilon)) * np.array(grad_w, dtype=np.float)) / (
                np.sqrt(epsilon + np.array(r2_w, dtype=np.float)))
                  for U_w, grad_w, r2_w in zip(self.U_weights, grad_weights, self.delta_weights)]
            dB = [((np.sqrt(np.array(U_b, dtype=np.float) + epsilon)) * np.array(grad_b, dtype=np.float)) / (
                np.sqrt(epsilon + np.array(r2_b, dtype=np.float)))
                  for U_b, grad_b, r2_b in zip(self.U_bias, grad_bias, self.delta_bias)]

            self.U_weights = [rho2 * U_w + (1 - rho2) * (np.power(dw, 2)) for U_w, dw in zip(self.U_weights, dW)]
            self.U_bias = [rho2 * B_w + (1 - rho2) * (np.power(db, 2)) for B_w, db in zip(self.U_bias, dB)]

        elif update_rule == "adam":
            rho1 = 0.9
            rho2 = 0.999
            # u = sum of values for g(w)
            self.delta_weights = [rho1 * u_w + (1 - rho1) * g_w for u_w, g_w in zip(self.delta_weights, grad_weights)]
            self.delta_bias = [rho1 * u_b + (1 - rho1) * g_b for u_b, g_b in zip(self.delta_bias, grad_bias)]

            # v = sum of values of g(w)^2
            self.U_weights = [rho2 * v_w + (1 - rho2) * (np.power(g_w, 2)) for v_w, g_w in
                              zip(self.U_weights, grad_weights)]
            self.U_bias = [rho2 * v_b + (1 - rho2) * (np.power(g_b, 2)) for v_b, g_b in zip(self.U_bias, grad_bias)]

            # u_hat = u/ (1-rho1)^m
            self.delta_weights_hat = self.delta_weights / (1 - np.power(rho1, self.epoch_num))
            self.delta_bias_hat = self.delta_bias / (1 - np.power(rho1, self.epoch_num))

            # v_hat = v/ (1-rho2)^m
            self.U_weights_hat = self.U_weights / (1 - np.power(rho2, self.epoch_num))
            self.U_bias_hat = self.U_bias / (1 - np.power(rho2, self.epoch_num))

            self.weights = [
                w_old - ((eta * np.array(u_hat_w, dtype=np.float)) / (
                    np.sqrt(epsilon + np.array(v_hat_w, dtype=np.float))))
                for w_old, u_hat_w, v_hat_w in zip(self.weights, self.delta_weights_hat, self.U_weights_hat)]
            self.bias = [
                b_old - ((eta * np.array(u_hat_b, dtype=np.float)) / (
                    np.sqrt(epsilon + np.array(v_hat_b, dtype=np.float))))
                for b_old, u_hat_b, v_hat_b in zip(self.bias, self.delta_bias_hat, self.U_bias_hat)]

        predicted, error = self.test(x, y)
        return error

    def update_weights_stochastic(self, x, y, N, eta, update_rule="delta", alpha=.9, epsilon=1e-6):
        # N : size of training data
        # x : Input Data (features)
        # y : desired output

        # calculating gradients by seeing each example and updating the weights

        for i in range(N):
            curr_grad_bias, curr_grad_weights = self.__back_propagation__(x[i], y[i])
            self.epoch_num += 1
            # update the weights and bias
            if update_rule == "delta":
                self.weights = [w_old - eta * grad_w for w_old, grad_w in zip(self.weights, curr_grad_weights)]
                self.bias = [b_old - eta * grad_b for b_old, grad_b in zip(self.bias, curr_grad_bias)]

            elif update_rule == "general_delta":
                self.weights = [w_old - eta * grad_w + alpha * del_w for w_old, grad_w, del_w in
                                zip(self.weights, curr_grad_weights, self.delta_weights)]
                self.bias = [b_old - eta * grad_b + alpha * del_b for b_old, grad_b, del_b in
                             zip(self.bias, curr_grad_bias, self.delta_bias)]
                self.delta_weights = [-eta * g_w + alpha * del_w for g_w, del_w in
                                      zip(curr_grad_weights, self.delta_weights)]
                self.delta_bias = [-eta * g_w + alpha * del_w for g_w, del_w in zip(curr_grad_bias, self.delta_bias)]

            elif update_rule == "ada_grad":
                # updating the parameters using the learning rate for each parameter  by adagrad
                #  updating r for next iteration, i.e. accumulating squares of gradients for next iteration
                self.delta_weights = [r2_w + (np.power(g_w, 2)) for r2_w, g_w in
                                      zip(self.delta_weights, curr_grad_weights)]
                self.delta_bias = [r2_b + (np.power(g_b, 2)) for r2_b, g_b in zip(self.delta_bias, curr_grad_bias)]

                self.weights = [w_old - (eta * np.array(grad_w, dtype=np.float)) / (
                    np.sqrt(epsilon + np.array(r2_w, dtype=np.float)))
                                for w_old, grad_w, r2_w in zip(self.weights, curr_grad_weights, self.delta_weights)]
                self.bias = [b_old - (eta * np.array(grad_b, dtype=np.float)) / (
                    np.sqrt(epsilon + np.array(r2_b, dtype=np.float)))
                             for b_old, grad_b, r2_b in zip(self.bias, curr_grad_bias, self.delta_bias)]

            elif update_rule == "rmsProp":
                rho = 0.95
                self.delta_weights = [rho * r2_w + (1 - rho) * (np.power(g_w, 2)) for r2_w, g_w in
                                      zip(self.delta_weights, curr_grad_weights)]
                self.delta_bias = [rho * r2_b + (1 - rho) * (np.power(g_b, 2)) for r2_b, g_b in
                                   zip(self.delta_bias, curr_grad_bias)]

                self.weights = [
                    w_old - ((eta * np.array(grad_w, dtype=np.float)) / (
                        np.sqrt(epsilon + np.array(r2_w, dtype=np.float))))
                    for w_old, grad_w, r2_w in zip(self.weights, curr_grad_weights, self.delta_weights)]
                self.bias = [
                    b_old - (eta * np.array(grad_b, dtype=np.float)) / (
                        np.sqrt(epsilon + np.array(r2_b, dtype=np.float)))
                    for b_old, grad_b, r2_b in zip(self.bias, curr_grad_bias, self.delta_bias)]

            elif update_rule == "adaDelta":
                rho1 = 0.95
                rho2 = 0.9

                # rms values used in denominator same as Ada Grad
                self.delta_weights = [rho1 * r2_w + (1 - rho1) * (np.power(g_w, 2)) for r2_w, g_w in
                                      zip(self.delta_weights, curr_grad_weights)]
                self.delta_bias = [rho1 * r2_b + (1 - rho1) * (np.power(g_b, 2)) for r2_b, g_b in
                                   zip(self.delta_bias, curr_grad_bias)]

                self.weights = [
                    w_old - ((np.sqrt(np.array(U_w, dtype=np.float) + epsilon)) * np.array(grad_w, dtype=np.float)) / (
                        np.sqrt(epsilon + np.array(r2_w, dtype=np.float)))

                    for w_old, U_w, grad_w, r2_w in
                    zip(self.weights, self.U_weights, curr_grad_weights, self.delta_weights)]

                self.bias = [
                    b_old - ((np.sqrt(np.array(U_b, dtype=np.float) + epsilon)) * np.array(grad_b, dtype=np.float)) / (
                        np.sqrt(epsilon + np.array(r2_b, dtype=np.float)))
                    for b_old, U_b, grad_b, r2_b in zip(self.bias, self.U_bias, curr_grad_bias, self.delta_bias)]

                # Update U for the next iteration
                dW = [((np.sqrt(np.array(U_w, dtype=np.float) + epsilon)) * np.array(grad_w, dtype=np.float)) / (
                    np.sqrt(epsilon + np.array(r2_w, dtype=np.float)))
                      for U_w, grad_w, r2_w in zip(self.U_weights, curr_grad_weights, self.delta_weights)]

                dB = [((np.sqrt(np.array(U_b, dtype=np.float) + epsilon)) * np.array(grad_b, dtype=np.float)) / (
                    np.sqrt(epsilon + np.array(r2_b, dtype=np.float)))
                      for U_b, grad_b, r2_b in zip(self.U_bias, curr_grad_bias, self.delta_bias)]

                self.U_weights = [rho2 * U_w + (1 - rho2) * (np.power(dw, 2)) for U_w, dw in zip(self.U_weights, dW)]
                self.U_bias = [rho2 * B_w + (1 - rho2) * (np.power(db, 2)) for B_w, db in zip(self.U_bias, dB)]

            elif update_rule == "adam":
                rho1 = 0.95
                rho2 = 0.9

                # u = sum of values for g(w)
                self.delta_weights = [rho1 * u_w + (1 - rho1) * g_w for u_w, g_w in
                                      zip(self.delta_weights, curr_grad_weights)]
                self.delta_bias = [rho1 * u_b + (1 - rho1) * g_b for u_b, g_b in zip(self.delta_bias, curr_grad_bias)]

                # v = sum of values of g(w)^2
                self.U_weights = [rho2 * v_w + (1 - rho2) * (np.power(g_w, 2)) for v_w, g_w in
                                  zip(self.U_weights, curr_grad_weights)]
                self.U_bias = [rho2 * v_b + (1 - rho2) * (np.power(g_b, 2)) for v_b, g_b in
                               zip(self.U_bias, curr_grad_bias)]

                # u_hat = u/ (1-rho1)^m
                self.delta_weights_hat = self.delta_weights / (1 - np.power(rho1, self.epoch_num))
                self.delta_bias_hat = self.delta_bias / (1 - np.power(rho1, self.epoch_num))

                # v_hat = v/ (1-rho2)^m
                self.U_weights_hat = self.U_weights / (1 - np.power(rho2, self.epoch_num))
                self.U_bias_hat = self.U_bias / (1 - np.power(rho2, self.epoch_num))

                self.weights = [
                    w_old - ((eta * np.array(u_hat_w, dtype=np.float)) / (
                        np.sqrt(epsilon + np.array(v_hat_w, dtype=np.float))))
                    for w_old, u_hat_w, v_hat_w in zip(self.weights, self.delta_weights_hat, self.U_weights_hat)]

                self.bias = [
                    b_old - ((eta * np.array(u_hat_b, dtype=np.float)) / (
                        np.sqrt(epsilon + np.array(v_hat_b, dtype=np.float))))
                    for b_old, u_hat_b, v_hat_b in zip(self.bias, self.delta_bias_hat, self.U_bias_hat)]

    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias


    def test(self, x, y):
        n = len(y)
        predicted_y = []
        for i in range(n):
            predicted_y.append(self.__forward_pass__(x[i]))
        diff = np.array(predicted_y) - np.array(y)
        diff = np.power(diff, 2)
        error = 0.5*np.sum(np.sum(diff, axis=0)) / n
        return predicted_y, error
