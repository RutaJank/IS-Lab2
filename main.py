import numpy as np
import helper as h
import matplotlib.pyplot as plt
np.random.seed(4)
d = []
x = np.linspace(0.1, 1, 20)
d = h.desired_fromula(x)

w1 = np.random.rand(5)
b1 = np.random.rand(5)

w2 = np.random.rand(5)
b2 = np.random.rand()
v1 = np.empty(5)
y1 = np.empty(5)
v2 = 0
e = 0
delta = np.empty(5)

p = 0.1

for i in range(10000):
    for j in range(len(x)):
        v1 = h.get_parameter_value(x[j],w1,b1)
        y1 = h.sigmoid(v1)
        v2 = h.get_layer2_parameter_value(y1, w2, b2)
        e = d[j] - v2
        delta = y1 * (1-y1) * (e * w2)
        w2 += p * e * y1
        w1 += p * delta * x[j]
        b1 += p * delta
        b2 += p * e

d2 = []
for i in range(len(x)):
    v1 = h.get_parameter_value(x[i],w1,b1)
    y1 = h.sigmoid(v1)
    d2.append(h.get_layer2_parameter_value(y1, w2, b2))

fig, ax = plt.subplots()
ax.plot(x, d, x, d2)
plt.show()