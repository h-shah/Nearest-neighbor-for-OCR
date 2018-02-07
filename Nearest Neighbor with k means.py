# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 20:36:40 2016

@author: hshah
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import random
from sklearn.cluster import KMeans
from scipy.io import loadmat


def main():
    ocr = loadmat('ocr.mat')
    
    means=[]
    stddev=[]
    Ns= [1000, 2000, 4000, 8000]
    
    for n in Ns:
        results=[]
        for i in range(0,10):
            print n
            print i
            kmeans = KMeans(n_clusters=n, init='random', n_init=1, max_iter=3, precompute_distances=True, verbose=0)
            kmeans.fit_predict(ocr['data']).astype('float')
            trainingFeatures = kmeans.cluster_centers_
            sp.stats.mode(ocr['labels'][kmeans.labels_ == 7], axis=0)[0]
            trainingLabels = np.empty(shape=(n,1))
            for i in range(0,n):
                trainingLabels[i] = sp.stats.mode(ocr['labels'][kmeans.labels_ == i], axis=0)[0]
            preds=NN1(trainingFeatures,trainingLabels,ocr['testdata'].astype('float'))
            errorRate = float(np.count_nonzero(preds != ocr['testlabels']))/ float(10000)
            results.append(errorRate)
        means.append(np.mean(results))
        stddev.append(np.std(results))
    print means
    print stddev
    plt.errorbar(Ns,means,yerr=stddev)
    plt.xticks(xrange(0,10000,1000))
    plt.xlabel("training sample size")
    plt.ylabel("error rate, 1 std dev bars")
    plt.savefig('graph2.jpeg')
        
    
def NN1(X,Y,test):
    X_squared= np.sum(np.square(X),axis=1)[:,np.newaxis]
    X_times_test=np.dot(X,test.T)*2
    SquaredDiff=X_squared-X_times_test
    preds= Y[np.argmin(SquaredDiff,axis=0)]
    return preds

if __name__ == "__main__":
    main()
    