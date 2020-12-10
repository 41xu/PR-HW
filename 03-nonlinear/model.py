import numpy as np


class KernelFisherDiscriminant():
    def __init__(self):
        self.alpha = None
        self.gamma = None
        self.N = None
        self.M = None
        self.x=None
    def fit(self, x_train, y_train):
        self.gamma = 1. / x_train.shape[1] * np.var(x_train)
        index1 = np.array([index for index, value in enumerate(y_train) if value == 1])
        index2 = np.array([index for index, value in enumerate(y_train) if value == 2])
        x1 = x_train[index1]
        x2 = x_train[index2]
        self.x=np.vstack((x1,x2))
        k1 = np.zeros((x_train.shape[0], x1.shape[0]))
        k2 = np.zeros((x_train.shape[0], x2.shape[0]))
        self.M = np.zeros((x_train.shape[0], 1))

        for i in range(x_train.shape[0]):
            k1[i] = self.rbf(np.tile(self.x[i], (x1.shape[0], 1)), x1, self.gamma)
            k2[i] = self.rbf(np.tile(self.x[i], (x2.shape[0], 1)), x2, self.gamma)
            self.M[i] = sum(k1[i]) / x1.shape[0] - sum(k2[i]) / x2.shape[0]
        self.N = (1 - 1. / x1.shape[0]) * k1.dot(k1.T) + (1 - 1. / x2.shape[0]) * k2.dot(k2.T)
        self.alpha = np.linalg.pinv(self.N).dot(self.M)

    def rbf(self, x, y, gamma):
        return np.exp(-gamma * np.square(x - y).sum(axis=1))

    def predict(self, x_test):
        return self.alpha.T.dot(self.rbf(self.x, np.tile(x_test, (self.x.shape[0], 1)), self.gamma).reshape(-1, 1))

    def score(self, x_test, y_test):
        tmp = [1 if self.predict(x) >= 0 else 2 for x in x_test]
        s = np.sum(tmp == y_test)
        return s / x_test.shape[0]
