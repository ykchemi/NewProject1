import numpy
import matplotlib.pyplot as plt
import tqdm

sigma = numpy.zeros((2, 4))

x = numpy.array([[0, 0], [0, 1], [1, 0], [1, 1]])
t = numpy.array([1, 1, 1, 0])

u = numpy.zeros(2)
b = numpy.zeros(2)

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

for hoge in tqdm.tqdm(range(100000)):
    for i, e in enumerate(x):
        u[1] = numpy.dot(e, w1) + b[1]
        y1.append(f(u[1]))
        u = f(u[1]) * w + b
        y.append(f(u[0]))
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
            sigma[0] = numpy.array(y) - t
            sigma[1] = numpy.array([sigma[0, m]*w*Derivative(sigma[1, m]) for m in range(4)])
            w = w - eta * (numpy.sum(numpy.array([sigma[0, m]*x[m, 0] for m in range(4)]))/4)
            b[0] = b[0] - eta * (numpy.sum(sigma[0])/4)
            for nyan in range(len(w1)):
                w1[nyan] = w1[nyan] - eta * (numpy.sum(numpy.array([sigma[1, m]*x[m, nyan] for m in range(4)]))/4)
            b[1] = b[1] - eta * (numpy.sum(sigma)/4)

            #---for graph---
            #graph_y.append(BA(y, t))
            graph_y.append(y)

            y = []
            y1 = []

print(y)

plt.plot(numpy.arange(len(graph_y)), numpy.array(graph_y))
plt.show()
