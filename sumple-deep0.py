import numpy
import matplotlib.pyplot as plt
import tqdm

sigma = numpy.zeros(4)
sigma1 = numpy.zeros(4)

x = numpy.array([[0, 0], [0, 1], [1, 0], [1, 1]])
t = numpy.array([0, 1, 1, 1])

u = 0
b = 0
u1 = 0
b1 = 0

w = 0
w1 = numpy.zeros(2)
y = []
y1 = []

eta = 0.7

def f(u):
    #return 1 / (1 + 2**-1*u)
    return max((0, u))

def Derivative(x):
    if x > -1:
        return 1
    else:
        return 0

def BA(y, t):
    a = y - t
    return numpy.sum(a)/4

graph_y = []

for hoge in tqdm.tqdm(range(10000)):
    for i, e in enumerate(x):
        u1 = numpy.dot(e, w1) + b1
        y1.append(f(u1))
        u = f(u1) * w + b
        y.append(f(u))
        '''
        print(f(u))
        if f(u) > 0.85:
            y.append(1)
            sigma[i] = 1 - t[i]
        else:
            y.append(0)
            sigma[i] = 0 - t[i]
        '''
        if i == 3:
            sigma = numpy.array(y) - t
            sigma1 = numpy.array([sigma[m]*w*Derivative(sigma1[m]) for m in range(4)])
            w = w - eta * (numpy.sum(numpy.array([sigma[m]*x[m, 0] for m in range(4)]))/4)
            b = b - eta * (numpy.sum(sigma)/4)
            w1[0] = w1[0] - eta * (numpy.sum(numpy.array([sigma1[m]*x[m, 0] for m in range(4)]))/4)
            w1[1] = w1[1] - eta * (numpy.sum(numpy.array([sigma1[m]*x[m, 1] for m in range(4)]))/4)
            b1 = b1 - eta * (numpy.sum(sigma1)/4)

            #---for graph---
            #graph_y.append(BA(y, t))
            graph_y.append(y)

            y = []
            y1 = []

plt.plot(numpy.arange(len(graph_y)), numpy.array(graph_y))
plt.show()
