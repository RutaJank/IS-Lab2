import numpy as np

def get_parameter_value(x, w, b):
    v = x*w + b
    return v

def get_layer2_parameter_value(y, w, b):
    v2 = sum(np.multiply(y, w)) + b
    return v2

def desired_fromula(x):
    y = (1 + 0.6 * np.sin(2 * np.pi * x / 0.7)) + 0.3 * np.sin(2 * np.pi * x)/ 2
    return y

def sigmoid(v):
    y = 1/(1+np.exp(-1*v))
    return y