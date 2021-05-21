import numpy

sigma = numpy.zeros(4)

x = numpy.array([[0, 0], [0, 1], [1, 0], [1, 1]])
t = numpy.array([0, 1, 1, 0])

u = 0
u1 = 0
u2 = 0
b = 0
b1 = 0
b2 = 0

w = numpy.zeros(2)
w1 = numpy.zeros(2)
w2 = numpy.zeros(2)
u1 = []
u2 = []
y = []
y1 = []
y2 = []
eta = 0.5

def f(u):
    #return 1 / (1 + 2**-1*u)
    return max((0, u))

for hoge in range(10):
    for i, e in enumerate(x):
        u = numpy.dot(w, e) + b
        print(f(u))
        '''
        memo
        if f(u) > 0:
            y.append(1)
            ここにsigma[i] = 1 - t[i]みたいなのは許容できない。
            パラメーターが狂う可能性がある。
        else:
            y.append(0)
        '''
        y.append(f(u))

        if i == 3:
            sigma = numpy.array(y) - t
            w[0] = w[0] - eta * (numpy.sum(numpy.array([sigma[m]*x[m, 0] for m in range(4)])))
            w[1] = w[1] - eta * (numpy.sum(numpy.array([sigma[m]*x[m, 1] for m in range(4)])))
            b = b - eta * (numpy.sum(sigma)/4)
            print(y)
            y = []
