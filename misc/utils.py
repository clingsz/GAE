# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 13:34:52 2016

@author: cling
"""
import numpy
import cPickle,gzip
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import spearmanr
from sklearn import linear_model

def savefig(fileName):
    plt.savefig(fileName, bbox_inches='tight')


def save_csv_table(fname,x,col_name=None,row_name=None):
    m,n = x.shape
    if row_name is None:
        row_name = ['r'+str(i) for i in range(m)]
    if col_name is None:
        col_name = ['c'+str(i) for i in range(n)]
#    print m,n
#    print len(row_name),len(col_name)
    print type(x[0,0])
    if type(x[0,0]) is numpy.string_:
        isstring = True
    else:
        isstring = False
    with open(fname,'wb') as f:
        f.write(' ,')
        for i in range(n-1):
            f.write(col_name[i] + ',')
        f.write(col_name[n-1] + '\n')
        for j in range(m):
            f.write(row_name[j]+',')
            for i in range(n-1):
                if isstring:
                    f.write(('%s,' % (x[j,i])))
                else:
                    f.write(('%.2f,' % (x[j,i])))
            if isstring:
                f.write(('%s\n' % (x[j,n-1])))
            else:
                f.write(('%.2f\n' % (x[j,n-1])))
#            f.write(('%.2f\n' % (x[j,n-1])))
    print fname, ' saved!'

def write_a_list(f,testgenes):
        f.write(testgenes[0])
        for g in testgenes[1:]:
            f.write(',' + g)
        f.write('\n')

def reorder(x,lst):
    y = []
    for i in lst:
        y.append(x[i])
    return y
        
    
def regress_out(x,y):
    regr = linear_model.LinearRegression()
    regr.fit(x,y)
    y_res = y - regr.predict(x)    
    return y_res

def linear_distance(x,y):
    y_res = regress_out(x,y)    
    dis = numpy.var(y_res)/numpy.var(y)  
#    print numpy.var(y_res),numpy.var(y)
    return dis
    

def corr(x,y):
#    C,b = spearmanr(numpy.concatenate([x,y],axis=1))
    if x.ndim==1:
        x = numpy.reshape(x,[len(x),1])
    if y.ndim==1:
        y = numpy.reshape(y,[len(y),1])

    C = numpy.corrcoef(numpy.concatenate([x,y],axis=1).transpose())
    p1 = x.shape[1]
    return C[:p1,p1:]

def spearmancorr(x,y):
    xy = numpy.concatenate([x,y],axis=1)
    C,b = spearmanr(xy)
    if xy.shape[1]==2:
        return C,b
    else:
        p1 = x.shape[1]
        return C[:p1,p1:],b[:p1,p1:]
    

def bart(x,xlbs=None):
    p = len(x)
    if xlbs is None:
        xlbs = range(p)
    plt.bar(numpy.arange(p)+0.5,x)
    plt.xticks(numpy.arange(p)+1,xlbs,rotation='vertical')
    plt.grid()

def get_colors(n):
    cm_subsection = numpy.linspace(0, 1, n) 
    colors = [ cm.viridis(x) for x in cm_subsection ]
    return colors
    
def mse(x,px):
    return numpy.mean((x-px)*(x-px))

def mae(x,px):
    return numpy.mean(numpy.abs(x-px))
    
def imagesc(x):
    plt.imshow(x,aspect='auto',interpolation='none')

def unit_vector(vector):
    return vector / numpy.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return numpy.arccos(numpy.clip(numpy.dot(v1_u, v2_u), -1.0, 1.0))

def saveobj(filename='temp.pkl.gz',obj=[]):
    fp=gzip.open(filename,'wb')
    cPickle.dump(obj,fp)
    print filename + ' saved!'

def loadobj(filename):
    print filename + ' loading...'
    fp=gzip.open(filename,'rb')    
    return cPickle.load(fp)
  
def get_total(lsts):
    TO = []
    for x in lsts:
        TO.append(len(x))
    total = int(numpy.prod(TO))
    print total
    
def getJobOpts(id,lsts):
    TO = []
    for x in lsts:
        TO.append(len(x))
    total = int(numpy.prod(TO))
    #   print total
    temp = id
    out = []
    for i in range(0,len(lsts)):
        mult = numpy.prod(TO[i+1:])
        indx = int(numpy.floor(temp/mult))
        temp = temp - indx*mult
        out.append(lsts[i][indx])
    return out

# generate k random samples under particular distribution p, with temprature T
def genSampleWithDistribution(p,k,rng,T=0):
#    rng = numpy.random.RandomState(rngid)
    p = p + T
    p = p*1.0/numpy.sum(p)
    lst = []
    v = numpy.sort(rng.rand(k))
    j = 0
    nowp = 0
    i = 0
    while j<k:
        nowv = v[j]
        if nowp<=nowv and nowp+p[i]>nowv:
            lst.append(i)
            j = j+1
        else:
            nowp = nowp + p[i]
            i = i+1
    return lst
