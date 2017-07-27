# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 14:19:29 2017

@author: cling
"""
from sklearn.cluster import AgglomerativeClustering as ag
import numpy
import matplotlib.pyplot as plt
import misc.gap as gap
###########################
# Clustering main methods
###########################

def gap_cluster(x):
    best_k = gap.fit_gap_stats(x,bootstraps=100,kMin=1,kMax=100)
    cids,counts,mses = ag_clust(x,best_k)
    return (cids,counts,mses)
    
def ag_clust(x,k):
    a = ag(n_clusters=k)
    a.fit(x)
    lbs = a.labels_
    cids = []
    for i in range(k):
        lst = numpy.where(lbs==i)[0]
        cids.append(lst)
    counts,mses = analyze_cluster(x,cids)
    lst = get_order(mses)
    mses = reorder(mses,lst)
    cids = reorder(cids,lst)
    counts = reorder(counts,lst)    
    show_cluster(x,cids,mses)
    return cids,counts,mses
    
def analyze_cluster(x,cids):
    counts = []
    mses = []
    for i in range(len(cids)):
        y = x[cids[i],:]
        ym = numpy.mean(y,axis=0)
        ym = numpy.reshape(ym,[1,len(ym)])
        ya = numpy.repeat(ym,y.shape[0],axis=0)
        m = numpy.mean((y - ya)**2)
        print i,y.shape[0],m
        counts.append(y.shape[0])
        mses.append(m)
    return counts,mses

def reorder(A,lst):
    B = []
    for i in range(len(A)):
        B.append(A[lst[i]])
    return B

def get_order(metric):
    ke = metric
    lst = sorted(range(len(ke)),key=lambda x:ke[x])
    return lst    
####################################
# Clustering visualization methods
####################################
    
def draw_bound(bounds,x):
    nb = -0.5
    pos = []
    for b in bounds:        
        pos.append((nb+nb+b)/2)
        nb += b
        plt.plot([nb,nb],[-1,x.shape[0]],'k--',markerSize=10)
    plt.xlim([0-0.5,x.shape[1]-0.5])
    plt.ylim([0-0.5,x.shape[0]-0.5])
    return pos

def show_cluster(x,cids=None,mses=None,obNames=None):
    if cids is None:
        cids = [[i for i in range(x.shape[0])]]
    K = len(cids)
    x = x.transpose()
    bounds = []
    for i in range(K):
        bounds.append(len(cids[i]))
    plt.figure(figsize=[15,15])
    plt.imshow(x,aspect='auto',interpolation='none',vmax=3,vmin=-3,cmap='PRGn')
    plt.colorbar()
    if obNames is not None:
        plt.yticks(range(len(obNames)),obNames)
    pos = draw_bound(bounds,x)
    if mses is not None:
        mses = numpy.round(mses,decimals=2)
        plt.xticks(pos,mses,rotation='vertical')
    plt.xlabel('MSEs')
############## test #################
def test():
#    x = numpy.random.randn(100,8)
    x = gap.init_board_gauss(200,5)
    gap_cluster(x)
    