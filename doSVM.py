from numpy import array
import numpy as np
from getData import getData
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler 
from sklearn import svm
import matplotlib.pyplot as plt


def doSVM(features:array, labels: array):
    pca = PCA(n_components=2)
    scl = StandardScaler()
    features = scl.fit_transform(features)
    features = pca.fit_transform(features)
    clf = svm.SVC()
    clf.fit(features,labels)
    return clf





