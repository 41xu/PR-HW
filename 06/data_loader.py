import numpy as np
import matplotlib.pyplot as plt


# pth = './data.txt'


def load_data(pth):
    with open(pth) as f:
        lines = [list(map(float, line.split())) for line in f]
    x = np.array(lines)[:, 1:]
    y = np.array(lines)[:, 0].astype(int)
    return x, y


def visual(data, label,title):
    color = ['navy', 'turquoise', 'darkorange']
    for x, y in zip(data, label):
        plt.scatter(x[0],x[1],color=color[y-1])
        # plt.scatter(x[0],x[2],color=color[y-1])
        # plt.scatter(x[1], x[2], color=color[y - 1])
    plt.title(title)
    # plt.savefig('data12.png')
    plt.show()

#
# pth = './data.txt'
# x, y = load_data(pth)
# print(x, y)
# visual(x, y)
