import tqdm
import math
import numpy
import matplotlib.pyplot as plt
from PIL import Image

def f(x):
    return max((0, x))

def Derivative(l):
    returnl = []
    for i in l:
        if i > 0:
            returnl.append(1)
        else:
            returnl.append(0)
    return numpy.array(returnl)

x = numpy.array([[0, 0], [0, 1], [1, 0], [1, 1]])
t = numpy.array([0, 1, 1, 0])

b = numpy.random.random_sample(3)
w = numpy.random.random_sample((3, 2))
u = numpy.zeros((4, 3))
y = numpy.zeros((3, 4))
sigma = numpy.zeros((3, 4))
eta = 0.5

for i in range(100):
    for n, e in enumerate(x):
        u[n, 1] = numpy.dot(w[1, :], e) + b[1]
        u[n, 2] = numpy.dot(w[2, :], e) + b[2]
        u[n, 0] = numpy.dot(w[0, :], numpy.array([f(u[n, 1]), f(u[n, 2])])) + b[0]

        y[0, n] = u[n, 0]
        y[1, n] = u[n, 1]
        y[2, n] = u[n, 2]

        if n == 3:
            sigma[0, :] = numpy.array(y[0, :] - t)
            sigma[1, :] = sigma[0, :] * w[0, 0] * Derivative(u[:, 1])
            sigma[2, :] = sigma[0, :] * w[0, 1] * Derivative(u[:, 2])

            w[0, 0] = w[0, 0] - eta * (numpy.sum(sigma[0, :] * y[1, :]) / 4)
            w[0, 1] = w[0, 1] - eta * (numpy.sum(sigma[0, :] * y[2, :]) / 4)
            w[1, 0] = w[1, 0] - eta * (numpy.sum(sigma[1, :] * x[:, 0]) / 4)
            w[1, 1] = w[1, 1] - eta * (numpy.sum(sigma[1, :] * x[:, 1]) / 4)
            w[2, 0] = w[2, 0] - eta * (numpy.sum(sigma[2, :] * x[:, 0]) / 4)
            w[2, 1] = w[2, 0] - eta * (numpy.sum(sigma[1, :] * x[:, 1]) / 4)
            b[0] = b[0] - eta * (numpy.sum(sigma[0, :]) / 4)
            b[1] = b[1] - eta * (numpy.sum(sigma[1, :]) / 4)
            b[2] = b[2] - eta * (numpy.sum(sigma[2, :]) / 4)

print(w)
