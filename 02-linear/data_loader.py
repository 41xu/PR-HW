import numpy as np
import scipy.io as sio
from sklearn import model_selection

# pth = './data.mat'


def load_data(pth):
    data = sio.loadmat(pth)  # __header__, __version__, __globals__, data, data最后一列是label
    data = data['data']
    np.random.shuffle(data)
    x = data[:, :-1]
    y = data[:, -1]
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.3, random_state=0, stratify=y)
    return x_train, y_train, x_test, y_test

# load_data(pth)
