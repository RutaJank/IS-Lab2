import numpy as np
import helper as h
import matplotlib.pyplot as plt

x = np.linspace(-6, 6, 20)
y = np.linspace(-6, 6, 20)

X, Y = np.meshgrid(x, y)
D = h.get_desired_3d_function(X, Y)

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
        for n in range(len(x)):
            v1 = h.get_parameter_value_2(X[j][n],w1,Y[j][n],w1_2,b1)
            y1 = h.sigmoid(v1)
            v2 = h.get_layer2_parameter_value(y1, w2, b2)
            e = D[j][n] - v2
            delta = y1 * (1-y1) * (e * w2)
            w2 += p * e * y1
            w1 += p * delta * X[j][n]
            w1_2 += p * delta * Y[j][n]
            b1 += p * delta
            b2 += p * e


d2 = []
for i in range(len(x)):
    d2.append([])
    for j in range(len(x)):
        v1 = h.get_parameter_value_2(X[i][j],w1,Y[i][j],w1_2,b1)
        y1 = h.sigmoid(v1)
        d2[i].append(h.get_layer2_parameter_value(y1, w2, b2))



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, np.array(d2))
ax.plot_surface(X, Y, np.array(D))

plt.show()