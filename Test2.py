import sys
import math
import tqdm
import random
import numpy
#import pandas
from matplotlib import cm
import matplotlib.pyplot as plt
#from PIL import Image

#画像サイズ：50*50－＞10*10

plt.style.use('dark_background')
fig = plt.figure()

#Datasetpath = r'D:\ArduinoLearning\DATASET\FORSUMPLE\{}.png'

a = 0
#DataArray = numpy.zeros((10, 50, 50))

def f(u):
    return 1/(1+numpy.exp(-1*u))

f = numpy.vectorize(f)

def returnNewWn(Wn, index, _len):
    #return Wn-eta*(1/4)*numpy.sum(numpy.array([(y[i]-t[i])*Dataset[i, index] for i in range(_len)]))
    return Wn - eta * (1/4) * numpy.dot(sigma, Dataset[:, index])

graph_y = []
graph_y2 = []
graph_y3 = []

#x1, x2
Dataset = numpy.array([[0, 0], [1, 0], [0, 1], [1, 1]])
t = [0, 0, 0, 1]

#w1 = 0
#w2 = 0
w1 = random.random()
w2 = random.random()
eta = 0.1
c = 0
c2 = 0
#b = 0
b = random.random()
y = []

'''
for hoge in tqdm.tqdm(range(1000)):
    if c % 4 == 0 and c > 0:
        y = numpy.array(y)
        w1 = returnNewWn(w1, 0, 4)
        w2 = returnNewWn(w2, 1, 4)
        sigma = y - t
        #b = b-eta*(1/4)*numpy.sum(numpy.array([y[i]-t[i] for i in range(4)]))
        b = b - eta * (1/4) * numpy.sum(sigma)
        graph_y.append(numpy.count_nonzero(y == t))
        y = []
        c2 += 1
        graph_y.append(w1)
        graph_y2.append(w2)
    index = c - 4 * c2
    u = numpy.dot(Dataset[index], numpy.array([w1, w2]))+b
    y.append(f(u))
    c += 1
'''
backup_y = []
for hoge in tqdm.tqdm(range(10000)):
  '''
  for i in x:
    #index = c - 4 * c2
    #u = numpy.dot(Dataset[index], numpy.array([w1, w2]))+b
    u = numpy.dot(i, numpy.array([w1, w2]))+b
    y.append(f(u))
    #c += 1
    backup_y = y
  '''
  u = numpy.array([numpy.dot(numpy.array([w1, w2]), i)+b for i in Dataset])
  #y = numpy.array(y)
  y = f(u)
  sigma = y - numpy.array(t)
  w1 = returnNewWn(w1, 0, 4)
  w2 = returnNewWn(w2, 1, 4)
  #b = b-eta*(1/4)*numpy.sum(numpy.array([y[i]-t[i] for i in range(4)]))
  b = b - eta * (1/4) * numpy.sum(sigma)
  graph_y.append(numpy.count_nonzero(y == t))
  #y = []
  #c2 += 1
  graph_y.append(0)
  graph_y2.append(0)
  graph_y3.append(sum(sigma))

print('done')
#plt.plot(numpy.arange(len(graph_y)), numpy.array(graph_y))
ax1 = fig.add_subplot(1, 2, 2)
ax1.plot(numpy.arange(len(graph_y3)), numpy.array(graph_y3))

X = numpy.linspace(0, 1, 25)
Y = numpy.linspace(0, 1, 25)
X, Y = numpy.meshgrid(X, Y)
u = (X*w1)+(Y*w2)+b
Z = f(u)
ax = fig.add_subplot(1, 2, 1, projection='3d')
#surf = ax.plot_wireframe(X, Y, Z)
surf = ax.plot_surface(X, Y, Z, cmap=cm.cool)
plt.show()

print(w1)
print(w2)
print(y)
