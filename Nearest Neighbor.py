# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 20:36:40 2016

@author: hshah
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import random
from scipy.io import loadmat


def main():
    ocr = loadmat('ocr.mat')
    
    means=[]
    stddev=[]
    Ns= [1000, 2000, 4000, 8000]
    
    for n in Ns:
        results=[]
        for i in range(0,10):
            sel = random.sample(xrange(60000),n)
            trainingFeatures = ocr['data'][sel].astype('float')
            trainingLabels = ocr['labels'][sel].astype('float')
            preds=NN1(trainingFeatures,trainingLabels,ocr['testdata'].astype('float'))
            errorRate = float(np.count_nonzero(preds != ocr['testlabels']))/ float(10000)
            results.append(errorRate)
        #print results
        means.append(np.mean(results))
        stddev.append(np.std(results))
    plt.errorbar(Ns,means,yerr=stddev)
    plt.xticks(xrange(0,10000,1000))
    plt.xlabel("training sample size")
    plt.ylabel("error rate, 1 std dev bars")
    plt.savefig('graph1.jpeg')
        
    
    
def NN1(X,Y,test):
    X_squared= np.sum(np.square(X),axis=1)[:,np.newaxis]
    X_times_test=np.dot(X,test.T)*2
    SquaredDiff=X_squared-X_times_test
    preds= Y[np.argmin(SquaredDiff,axis=0)]
    return preds

if __name__ == "__main__":
    main()
    