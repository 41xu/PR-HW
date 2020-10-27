import numpy as np
from data_loader import load_data


class Fisher():
    def __init__(self):
        self.weight = None
        self.bias = None

    def fit(self, x_train, y_train):  # 已知两类
        index1 = np.array([index for index, value in enumerate(y_train) if value == 1])
        index2 = np.array([index for index, value in enumerate(y_train) if value == 2])
        x1 = x_train[index1]
        x2 = x_train[index2]
        m1 = np.mean(x1, axis=0).reshape(-1, 1).T
        m2 = np.mean(x2, axis=0).reshape(-1, 1).T
        m = np.mean(x_train, axis=0).reshape(-1, 1).T
        s1 = (x1 - m1).T.dot(x1 - m1)
        s2 = (x2 - m2).T.dot(x2 - m2)
        self.weight = np.matrix(s1 + s2).I.dot((m1 - m2).T)  # matrix.I 可逆矩阵的转置矩阵，这样就不用求了 .H 共轭转置矩阵，.A矩阵转为基本数组
        self.bias = -self.weight.T.dot(m.T)
        # self.weight.shape:(56,1), self.bias.shape:(1,1)

    def predict(self, x_test, y_test):
        score = 0
        tmp = [1 if x.dot(self.weight) + self.bias >= 0 else 2 for x in x_test]
        score += sum(tmp == y_test)
        return score / x_test.shape[0]


class Logistic():
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self._theta = None

    def _sigmoid(self, t):
        return 1. / (1. + np.exp(-t))

    def fit(self, x_train, y_train, eta=0.01, n_iters=1e4):

        def J(theta, x_b, y):
            y_hat = self._sigmoid(x_b.dot(theta))
            return - np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)) / len(y)

        def dJ(theta, x_b, y):
            return x_b.T.dot(self._sigmoid(x_b.dot(theta)) - y) / len(y)

        def gradient_descent(x_b, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):
            theta = initial_theta
            cur_iter = 0
            while cur_iter < n_iters:
                gradient = dJ(theta, x_b, y)
                last_theta = theta
                theta = theta - eta * gradient
                if abs(J(theta, x_b, y) - J(last_theta, x_b, y)) < epsilon:
                    break
                cur_iter += 1
            return theta

        x_b = np.hstack([np.ones((len(x_train), 1)), x_train])
        initial_theta = np.zeros(x_b.shape[1])
        self._theta = gradient_descent(x_b, y_train, initial_theta, eta, n_iters)
        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self

    def predict(self, X_predict):
        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        proba = self._sigmoid(X_b.dot(self._theta))
        return np.array(proba >= 0.5, dtype='int')


    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return np.sum(y_test==y_predict)/len(y_predict)
