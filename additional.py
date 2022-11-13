import numpy as np
import helper as h
import matplotlib.pyplot as plt

d = []
x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)

d = h.get_desired_3d_function(x, y)

w1 = np.random.rand(5)
w1_2 = np.random.rand(5)
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
        v1 = h.get_parameter_value_2(x[j],w1,y[j],w1_2,b1)
        y1 = h.sigmoid(v1)
        v2 = h.get_layer2_parameter_value(y1, w2, b2)
        e = d[j] - v2
        delta = y1 * (1-y1) * (e * w2)
        w2 += p * e * y1
        w1 += p * delta * x[j]
        w1_2 += p * delta * y[j]
        b1 += p * delta
        b2 += p * e


d2 = []
for i in range(len(x)):
    v1 = h.get_parameter_value_2(x[i],w1,y[i],w1_2,b1)
    y1 = h.sigmoid(v1)
    d2.append(h.get_layer2_parameter_value(y1, w2, b2))


X, Y = np.meshgrid(x, y)
Z = np.meshgrid(d2)
D = np.meshgrid(d)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, np.array(Z))
ax.plot_surface(X, Y, np.array(D))

plt.show()