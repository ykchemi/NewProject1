import tqdm
import math
import numpy
import matplotlib.pyplot as plt
from PIL import Image

size1 = 2500
size2 = 99

Datasetpath = r'D:\ArduinoLearning\DATASET\FORSUMPLE\{}.png'
Data = numpy.array([numpy.array(Image.open(Datasetpath.format(str(i))).convert('L')).flatten() for i in range(1, 100)])
t = numpy.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0] * 100)

Wait = numpy.zeros((2500))

def f(u):
    return 1/(1+numpy.exp(-1*u))

def returnNewWn(Wn, index, _len):
    return Wn-eta*(1/4)*numpy.sum(numpy.array([(y[i]-t[i])*Data[i, index] for i in range(_len)]))

eta = 1
c = 0
c2 = 0
b = 0
y = []

for hoge in tqdm.tqdm(range(100)):
    for i in Data:
        if c % 99 == 0 and c > 0:
            y = numpy.array(y)
            Wait = numpy.array([returnNewWn(e, i, size2) for i, e in enumerate(Wait)])
            b = b-eta*(1/4)*numpy.sum(numpy.array([y[i]-t[i] for i in range(4)]))
            y = []
            c2 += 1
        index = c - size2 * c2
        u = numpy.dot(i, numpy.array(Wait))+b
        y.append(f(u))
        c += 1

w = Wait.reshape([50, 50])

print(w)

plt.imshow(w)
plt.show()
