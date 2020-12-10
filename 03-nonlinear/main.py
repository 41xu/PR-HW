import numpy as np
from data_loader import load_data
from model import KernelFisherDiscriminant
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

pth = './data.mat'

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_data(pth)
    # Neural Network
    mlp = MLPClassifier()
    mlp.fit(x_train, y_train)
    print("Neural Network score: %.4f" % mlp.score(x_test, y_test))

    # Non-Linear SVM
    nsvm = SVC(kernel='rbf', gamma='scale')
    nsvm.fit(x_train, y_train)
    print("Non-Linear SVM score: %.4f" % nsvm.score(x_test, y_test))

    # Kernel Fisher Discriminant
    kfd = KernelFisherDiscriminant()
    kfd.fit(x_train, y_train)
    print("Kernel Fisher  score: %.4f" % kfd.score(x_test, y_test))
