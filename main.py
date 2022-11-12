import numpy as np

def desired_fromula(x):
    return (1 + 0.6 * np.sin(2 * np.pi * x / 0.7)) + 0.3 * np.sin(2 * np.pi * x)/ 2

x = np.random.uniform(0, 1, 10)
d = desired_fromula(x)

w1 = np.random.uniform(0, 1, 5)
b1 = np.random.uniform(0, 1, 5)

w2 = np.random.uniform(0, 1, 5)
b2 = np.random.uniform(0, 1)

p = 0.1