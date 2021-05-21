import math
import sympy
import random
import tqdm
import numpy
from matplotlib import cm
import matplotlib.pyplot as plt

fig = plt.figure()

Data_length = 50
Epoc_value = 10000

border = 0.8
#x = numpy.random.random_sample((Data_length, 2))
x = numpy.random.random_sample((Data_length, 2))
t = numpy.zeros(Data_length)
s_x = numpy.array([numpy.dot(i[0], i[1]) for i in x])
print(numpy.where(s_x > border))
x1 = x[numpy.where(s_x > border), :]
x2 = x[numpy.where(s_x < border + 0.01), :]
t[numpy.where(s_x > border)] = 1
t[numpy.where(s_x < border + 0.01)] = 0
'''
x = numpy.array([[0, 0], [0, 1], [1, 0], [1, 1]])
t = numpy.array([0, 0, 0, 1])
'''
#w = numpy.random.randint(0, 10, 2)
#print(w)
w = numpy.random.random_sample(2)
#w = numpy.zeros(2)
#b = random.random()
b = random.randint(-10, 10)
y = numpy.zeros(Data_length)
sigma = numpy.zeros(Data_length)

graph_y = numpy.zeros((Data_length, Epoc_value))
graph_y2 = []

eta = 0.5

def f(x):
    exp = 2.718281828
    return 1 / (1+exp**-x)
f = numpy.vectorize(f)

def judge(i):
  return [1 if i > 0.5 else 0 for i in range(1)][0]

judge = numpy.vectorize(judge)

def returnNewWn(Wn, index, _len):
    #return Wn-eta*(1/Data_length)*numpy.sum(numpy.array([(y[i]-t[i])*x[i, index] for i in range(_len)]))
    return Wn - eta * (1/Data_length) * numpy.dot(sigma, x[:, index])

for n in tqdm.tqdm(range(Epoc_value)):
    y = numpy.array([f(numpy.dot(i, w)+b) for i in x])
    #graph_y[:, n] = numpy.array([numpy.dot(i, w) for i in x])
    graph_y[:, n] = y
    graph_y2.append(sum(y-t))
    sigma = y - t
    #print(sum(sigma))
    #w = w - eta * (numpy.array([numpy.dot(sigma, x[:, i]) for i in range(2)]) * (1/Data_length))
    #w[0] = w[0] - eta * (numpy.dot(sigma, x[:, 0]) / Data_length)
    w[0] = returnNewWn(w[0], 0, Data_length)
    #w[1] = w[1] - eta * (numpy.dot(sigma, x[:, 1]) / Data_length)
    w[1] = returnNewWn(w[1], 1, Data_length)
    b = b - eta * (numpy.sum(sigma) * (1/Data_length))
    '''
    if numpy.sum(sigma) < 0.01 and -0.01 < numpy.sum(sigma):
        print(y)
        break
    '''

X = numpy.linspace(0, 1, 25)
Y = numpy.linspace(0, 1, 25)
X, Y = numpy.meshgrid(X, Y)
u = (X*w[0])+(Y*w[1])+b
Z = f(u)


y1 = x1[0, :, 1]
x1 = x1[0, :, 0]
z1 = numpy.array([0.5 for i in range(len(x1))])
y2 = x2[0, :, 1]
x2 = x2[0, :, 0]
z2 = numpy.array([0.5 for i in range(len(x2))])

ax = fig.add_subplot(1, 2, 1, projection='3d')

ax.scatter(x1, y1, z1, c='red', s=3)
ax.scatter(x2, y2, z2, c='blue', s=3)

surf = ax.plot_wireframe(X, Y, Z)

ax2 = fig.add_subplot(1, 2, 2)
'''
for i in range(Data_length):
    ax2.plot(numpy.arange(Epoc_value), graph_y[i, :])
'''
ax2.plot(numpy.arange(Epoc_value), graph_y2)
plt.show()

print(y)
print(w)
print(b)
