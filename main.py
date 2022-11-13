import numpy as np
import helper as h
import matplotlib.pyplot as plt

d = []
x = np.linspace(0.1, 1, 10)
for  i in range(len(x)):
    d.append(h.desired_fromula(x[i]))

w1 = np.random.uniform(0, 1, 5)
b1 = np.random.uniform(0, 1, 5)

w2 = np.random.uniform(0, 1, 5)
b2 = np.random.uniform(0, 1)
v1 = np.empty(5)
y1 = np.empty(5)
v2 = 0
e = 0
delta = np.empty(5)

p = 0.1

for i in range(100000):
    for j in range(len(x)):
        for n in range(len(w1)):
            v1[n] = h.get_parameter_value(x[j],w1[n],b1[n])
        for n in range(len(v1)):
            y1[n] = h.sigmoid(v1[n])
        
        v2 = y1[0] * w2[0] + y1[1] * w2[1] + y1[2] * w2[2] + y1[3] * w2[3] + b2 

        e = d[j] - v2

        for n in range(len(y1)):
            delta[n] = y1[n] * (1-y1[n]) * (e * w2[n])
        
        for n in range(len(y1)):
            w2[n] += p * e * y1[n]
            w1[n] += p * delta[n] * x[j]
            b1 += p * delta[n]
        b2 += p * e
    
d2 = []
for i in range(len(x)):
    for n in range(len(w1)):
         v1[n] = h.get_parameter_value(x[i],w1[n],b1[n])
    for n in range(len(v1)):
        y1[n] = h.sigmoid(v1[n])
    d2.append(y1[0] * w2[0] + y1[1] * w2[1] + y1[2] * w2[2] + y1[3] * w2[3] + b2)

print(d)
print(d2)
fig, ax = plt.subplots()

ax.plot(x, d, x, d2)

plt.show()