import numpy
import matplotlib.pyplot as plt

x = numpy.array([[0, 0], [0, 1], [1, 0], [1, 1]])
t = numpy.array([0, 1, 1, 0])

sigma = numpy.zeros((3, 4))
eta = 0.5

u = numpy.zeros((3, 4))
b = numpy.zeros(3)

w = numpy.zeros((3, 2))

y = numpy.zeros((3, 4))

def f(u):
    return max((0, u))

def Derivative(x):
    if x > 0:
        return 1
    else:
        return 0

for j in range(1000):
    for i, e in enumerate(x):
        for n in range(1, 3):
            u[n, i] = numpy.dot(w[n], e) + b[n]
            y[n, i] = f(u[n])
        u[0, i] = numpy.dot(y[1:, i]) + b[0]
        y[0, i] = f(u[0])

        if i == 3:
            sigma[0] = y[0] - t
            w[0] = w[0] - eta * (numpy.array([numpy.dot(sigma[0], y[x]) / 4 for x in range(1, 3)]))])
            b[0] = b[0] - eta * (numpy.sum(sigma[0]) / 4)
            for n in range(1, 3):
                sigma[n] = sigma[0]*w[0, n]*u[n]
                w[n] = w[n] - eta * numpy.array([numpy.dot(sigma[m], x[:, m]) / 4 for m in range(1, 3)])
                b[n] = b[n] - eta * (numpy.sum(sigma[n]) / 4)
