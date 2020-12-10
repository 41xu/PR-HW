import matplotlib.pyplot as plt
import numpy as np

x = [_ for _ in range(-10, 11)]


def f1(x):
    y = [_ for _ in range(len(x))]
    for i in range(len(x)):
        y[i] = -1 / 24. * np.abs(x[i]) + 1. / 6 * np.abs(x[i] - 3) + 1. / 8 * np.abs(x[i] - 4)
    return y


def f2(x):
    y = [_ for _ in range(len(x))]
    for i in range(len(x)):
        y[i] = 1. / 6 * np.abs(x[i]) - 2. / 3 * np.abs(x[i] - 3) + 1. / 2 * np.abs(x[i] - 4)
    return y


def f3(x):
    y = [_ for _ in range(len(x))]

    for i in range(len(x)):
        y[i] = 1. / 8 * np.abs(x[i]) + 1. / 2 * np.abs(x[i] - 3) - 3. / 8 * np.abs(x[i] - 4)
    return y


y1 = f1(x)
y2=f2(x)
y3=f3(x)
plt.plot(x, y1)
plt.plot(x, y2)
plt.plot(x,y3)
plt.savefig('res.png')
plt.show()
