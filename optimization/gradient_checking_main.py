
# coding: utf-8

import numpy as np
from testCases_check import *
from gc_utils import sigmoid, relu, dictionary_to_vector, vector_to_dictionary, gradients_to_vector
from gradient_checking import *
x, theta = 2, 4
J = forward_propagation(x, theta)
print ("J = " + str(J))

x, theta = 2, 4
dtheta = backward_propagation(x, theta)
print ("dtheta = " + str(dtheta))

x, theta = 2, 4
difference = gradient_check(x, theta)
print("difference = " + str(difference))


X, Y, parameters = gradient_check_n_test_case()

cost, cache = forward_propagation_n(X, Y, parameters)
gradients = backward_propagation_n(X, Y, cache)
difference = gradient_check_n(parameters, gradients, X, Y)
