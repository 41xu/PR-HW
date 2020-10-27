import numpy as np
from data_loader import load_data
from model import Fisher
from sklearn.linear_model import LogisticRegression

pth = './data.mat'

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_data(pth)
    fisher = Fisher()
    fisher.fit(x_train, y_train)
    facc = fisher.predict(x_test, y_test)
    print("Fisher's linear discriminant score: %.2f" % float(facc * 100), "%")
    logist = LogisticRegression(solver='liblinear')
    logist.fit(x_train, y_train)
    lacc = logist.score(x_test, y_test)
    print("Logistic regression score: %.2f" % float(lacc * 100), '%')

