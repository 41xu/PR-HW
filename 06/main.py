import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import Isomap, LocallyLinearEmbedding, SpectralEmbedding
from data_loader import load_data


def visual(data, label, title):
    color = ['navy', 'turquoise', 'darkorange']
    for x, y in zip(data, label):
        plt.scatter(x[0], x[1], color=color[y - 1])
    plt.title(title)
    plt.savefig('{}.png'.format(title))
    plt.show()


if __name__ == '__main__':
    pth = './data.txt'
    data, label = load_data(pth)
    # print(data,label)
    # PCA
    pca = PCA(n_components=2)
    pca_ = pca.fit_transform(data)
    visual(pca_, label, "PCA")
    # LDA
    lda = LinearDiscriminantAnalysis()
    lda_ = lda.fit_transform(data, label)
    visual(lda_, label, "LDA")
    # KPCA
    kpca = KernelPCA(n_components=2, kernel='rbf')
    kpca_ = kpca.fit_transform(data)
    visual(kpca_, label, "KPCA")
    # Isomap
    iso = Isomap(n_components=2)
    iso_ = iso.fit_transform(data)
    visual(iso_, label, "Isomap")
    # LLE
    lle = LocallyLinearEmbedding(n_components=2)
    lle_ = lle.fit_transform(data)
    visual(lle_, label, "LLE")
    # Laplacian Eigenmaps
    le = SpectralEmbedding(n_components=2)
    le_ = le.fit_transform(data)
    visual(le_, label, "Laplacian Eigenmaps")
