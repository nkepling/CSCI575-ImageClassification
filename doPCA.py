from numpy import array
import numpy as np
from getData import getData
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt



class doPCA:
    """take pca"""
    def __init__(self) -> None:
        pass    
    def screePlot(features: array):
        pca = PCA()
        pca.fit(features)
        plt.figure()
        plt.plot(pca.explained_variance_ratio_, 'o-', color='blue')
        plt.title('Scree Plot')
        plt.xlabel("Number of Principal Components")
        plt.ylabel("Variance Explained")
        plt.show()


if __name__ == '__main__':
    X,y = getData.loadFeatureMat()
    doPCA.screePlot(X)