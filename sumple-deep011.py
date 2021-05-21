import math
import tqdm
import numpy
from matplotlib import cm
import matplotlib.pyplot as plt

x = numpy.array([[0, 0], [0, 1]])
t = numpy.array([0, 1])

#fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
fig = plt.figure()

eta = 1e-3

#b = numpy.zeros(3)
b = numpy.random.random_sample(4)

graph_y = []
graph_y21 = []
graph_y22 = []
graph_y23 = []
graph_y31 = []
graph_y32 = []
graph_y33 = []
graph_y34 = []
graph_y35 = []
graph_y36 = []
graph_y4 = []
graph_y5 = []

w = numpy.random.random_sample((3, 2))
OUT_w = numpy.random.random_sample(3)
#w = numpy.zeros((3, 2))
sigma = numpy.zeros((4, len(t)))
u = numpy.zeros((3, len(t)))

def f(x):
  #return max((0, x))
  return math.log(1+math.exp(x))

def f2(x):
    return 1/(1+numpy.exp(-1*x))

f = numpy.vectorize(f)
f2 = numpy.vectorize(f2)

def Derivative(x):
  #return [1 if x > 0 else 0 for i in range(1)][0]
  return math.exp(x)/(math.exp(x)+1)

def judge(i):
  return [1 if i > 0.75 else 0 for i in range(1)][0]

judge = numpy.vectorize(judge)

Derivative = numpy.vectorize(Derivative)

for i in tqdm.tqdm(range(10000)):
  u[0] = f(numpy.array([numpy.dot(w[0], x[i]) + b[0] for i in range(len(t))]))
  u[1] = f(numpy.array([numpy.dot(w[1], x[i]) + b[1] for i in range(len(t))]))
  u[2] = f(numpy.array([numpy.dot(w[2], x[i]) + b[2] for i in range(len(t))]))
  y = f2(numpy.array([numpy.dot(OUT_w, u[:, i]) + b[-1] for i in range(len(t))]))
  #y = judge(y)
  u[0] = numpy.where(u[0] == 0, 0.001, u[0])
  u[1] = numpy.where(u[1] == 0, 0.001, u[1])
  sigma[-1] = y - t
  sigma[0] = sigma[-1] * Derivative(u[0]) * OUT_w[0]
  sigma[1] = sigma[-1] * Derivative(u[1]) * OUT_w[1]
  sigma[2] = sigma[-1] * Derivative(u[2]) * OUT_w[2]
  OUT_w = OUT_w - eta * numpy.array([numpy.dot(sigma[-1], f(u[i]))/len(t) for i in range(3)])
  b[-1] = b[-1] - eta * (numpy.sum(sigma[-1])/len(t))
  for i in range(3):
    w[i] = w[i] - eta * numpy.array([numpy.dot(sigma[i], x[:, n])/len(t) for n in range(2)])
    b[i] = b[i] - eta * (numpy.sum(sigma[i])/len(t))
  graph_y.append(numpy.sum(sigma))
  graph_y21.append(b[0])
  graph_y22.append(b[1])
  graph_y23.append(b[2])
  graph_y31.append(w[0, 0])
  graph_y32.append(w[0, 1])
  graph_y33.append(w[1, 0])
  graph_y34.append(w[1, 1])
  graph_y35.append(w[2, 0])
  graph_y36.append(w[2, 1])
  graph_y4.append(sigma)
  graph_y5.append((len(numpy.where(y-t)[0])/len(t)))


#plt.plot(numpy.arange(len(graph_y)), numpy.array(graph_y))

'''
X = numpy.linspace(0, 2, 200)
Y = numpy.linspace(0, 2, 200)
X, Y = numpy.meshgrid(X, Y)
u = (X*w[0, 0])+(Y*w[0, 1]) + b[0]
Z = f(u)
ax = fig.add_subplot(2, 2, 1, projection='3d')
ax.set_title('Hidden1_NEURON')
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

X = numpy.linspace(0, 2, 100)
Y = numpy.linspace(0, 2, 100)
X, Y = numpy.meshgrid(X, Y)
u = (X*w[1, 0])+(Y*w[1, 1]) + b[1]
Z = f(u)
ax1 = fig.add_subplot(2, 2, 2, projection='3d')
ax1.set_title('Hidden2_NEURON')
surf = ax1.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

X = numpy.linspace(0, 2, 100)
Y = numpy.linspace(0, 2, 100)
X, Y = numpy.meshgrid(X, Y)
u = (X*w[2, 0])+(Y*w[2, 1]) + b[2]
Z = f(u)
ax2 = fig.add_subplot(2, 2, 3, projection='3d')
ax2.set_title('OUTPUT_NEURON')
surf = ax2.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

X = numpy.linspace(0, 2, 100)
Y = numpy.linspace(0, 2, 100)
X, Y = numpy.meshgrid(X, Y)
u1 = (X*w[0, 0])+(Y*w[0, 1]) + b[0]
u2 = (X*w[1, 0])+(Y*w[1, 1]) + b[1]
u = (f(u1*w[2, 0]))+f((u2*w[2, 1])) + b[2]
Z = f(u)
ax3 = fig.add_subplot(2, 2, 4, projection='3d')
ax3.set_title('OUT')
surf = ax3.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

plt.show()
'''

ax = fig.add_subplot(2, 3, 1)
ax.set_title('sum of sigma')
ax.plot(numpy.arange(len(graph_y)), graph_y)

ax2 = fig.add_subplot(2, 3, 2)
ax2.set_title('bias')
ax2.plot(numpy.arange(len(graph_y21)), graph_y21)
ax2.plot(numpy.arange(len(graph_y22)), graph_y22)
ax2.plot(numpy.arange(len(graph_y23)), graph_y23)

ax3 = fig.add_subplot(2, 3, 3)
ax3.set_title('wait')
ax3.plot(numpy.arange(len(graph_y31)), graph_y31)
ax3.plot(numpy.arange(len(graph_y32)), graph_y32)
ax3.plot(numpy.arange(len(graph_y33)), graph_y33)
ax3.plot(numpy.arange(len(graph_y34)), graph_y34)
ax3.plot(numpy.arange(len(graph_y35)), graph_y35)
ax3.plot(numpy.arange(len(graph_y36)), graph_y36)

X = numpy.linspace(0, 2, 100)
Y = numpy.linspace(0, 2, 100)
X, Y = numpy.meshgrid(X, Y)
u1 = (X*w[0, 0])+(Y*w[0, 1]) + b[0]
u2 = (X*w[1, 0])+(Y*w[1, 1]) + b[1]
u = (f(u1*w[2, 0]))+f((u2*w[2, 1])) + b[2]
Z = f(u)
ax3 = fig.add_subplot(2, 3, 4, projection='3d')
ax3.set_title('OUT')
surf = ax3.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

ax5 = fig.add_subplot(2, 3, 5)
ax5.set_title('Learn')
ax5.plot(numpy.arange(len(graph_y5)), graph_y5)

plt.show()

print(y)
