# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 10:57:23 2017

@author: Tianxiang Gao
"""

# learner

from gae.model.guidedae import GAEOpts,fitGAE
from sklearn.linear_model import ElasticNetCV,Ridge
import timeit
import numpy
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def PCA_train(x,n_components=None):
    if n_components is None:
        n_components = x.shape[1]
    else:
        n_components = min(n_components,x.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(x) 
    return pca.transform,pca.inverse_transform
    
def AE_train(x,y,width=2,depth=1,alpha=0.5,randseed=0):
    aeopt = GAEOpts()
    aeopt.w = width
    aeopt.d = depth
    aeopt.lam = alpha
    aeopt.randseed = randseed
    z = y
    time_start = timeit.default_timer()                    
    pred,encode,decode,epochs,predz = fitGAE(x,z,x,z,aeopt)
    used_time = timeit.default_timer() - time_start
    print 'Training used %ds, %d epochs' % (used_time,epochs)                                         
    return encode,decode

def var_exp(x,enc,dec):
    code = enc(x)
    x_new = dec(code)
    vx = numpy.var(x)
    vr = numpy.var(x-x_new)
    vxp = numpy.var(x_new)
#    print vx,vxp,vr
    r =  vxp/vx    
    return r
    
def visualizer(x,enc,dec,tit='',y=None):
    code = enc(x)
    r = var_exp(x,enc,dec)
    n,p = code.shape
    if y is None:
        plt.scatter(code[:,0],code[:,1])
    else:
        plt.scatter(code[:,0],code[:,1],c=y,cmap='hot',
                s=40,
                edgecolors='none',vmax=3,vmin=-3)
    plt.xlabel('dim 0')
    plt.ylabel('dim 1')
    plt.title('%s Variance Explained:%.2f%%' % (tit,r*100))
