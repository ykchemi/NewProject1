import tqdm
import math
import numpy
import matplotlib.pyplot as plt

size1 = 1000
size2 = 100

Data = numpy.random.random_sample((size1, size2))

t = numpy.array([1 if len(numpy.where(i>0.5)[0]) > 0 else 0 for i in Data])

Wait = numpy.zeros((size2))

def f(u):
    return 1/(1+math.exp(-1*u))

def returnNewWn(Wn, index, _len):
    return Wn-eta*(1/4)*numpy.sum(numpy.array([(y[i]-t[i])*Data[i, index] for i in range(_len)]))

eta = 0.5
c = 0
c2 = 0
b = 0
y = []

for hoge in tqdm.tqdm(range(size1**2)):
    if c % size1 == 0 and c > 0:
        y = numpy.array(y)
        Wait = numpy.array([returnNewWn(e, i, size1) for i, e in enumerate(Wait)])
        b = b-eta*(1/4)*numpy.sum(numpy.array([y[i]-t[i] for i in range(4)]))
        y = []
        c2 += 1
    index = c - size1 * c2
    u = numpy.dot(Data[index], numpy.array(Wait))+b
    y.append(f(u))
    c += 1

print(y)
