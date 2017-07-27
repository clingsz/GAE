# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 16:50:00 2017

@author: cling
"""

# making presentations


import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib

import numpy
from utils import loadobj,get_colors,spearmancorr,corr,bart
from data_gen import DataOpts,load_data,Combine_dataset
from exp_test import calc_best_test_err
from trainer import get_Jacobian, get_RCE_residual
import visualize as vis

def show_data_specification():
    dataopt = DataOpts(test_folds=5,valid_folds=3)
    C = Combine_dataset(dataopt)
    dsName = ['Montoya','Davis']
    for i in range(2):
        print dsName[i] + ' patients: ' + str(len(numpy.where(C.source==i)[0]))
    
    print 'Common Cytokines: ', len(C.x_labels)
    for c in C.x_labels:
        print c,
    print ''

def fig1_Raw_and_PCA_pred_error():
    ms,vs = vis.get_D5_ms_vs()
    w_lst = range(2,11)
    PRE_raw,PRE_pca,RCE_pca = vis.load_raw_pca_result(vs)        
    PRE_AE,RCE_AE = vis.load_ae_result(vs,attrid=[0])
    raw_m = numpy.mean(PRE_raw)
    pca_m = numpy.mean(PRE_pca,axis=1)
    pca_v = numpy.std(PRE_pca,axis=1)/5
    ae_m = numpy.mean(PRE_AE[:,0,:],axis=1)
    ae_v = numpy.std(PRE_AE[:,0,:],axis=1)/5
    plt.figure(figsize=[10,6])
    plt.errorbar(w_lst,pca_m,yerr=pca_v,label='PCA')
    plt.errorbar(w_lst,ae_m,yerr=ae_v,label='AE')
#    plt.errorbar(w2_lst,PRE_GAE_mean,yerr=gae_v,label='GAE')
    plt.plot([1,11],[raw_m]*2,'k--',label='RAW')
    plt.xlim([1,11])
    plt.legend()
    plt.xlabel('Code Length')
    plt.ylabel('Age Prediction Error')
    plt.show()

def fig2_gae_is_the_best_prediction():
    ms,vs = vis.get_D5_ms_vs()
    PRE_raw,PRE_pca,RCE_pca = vis.load_raw_pca_result(vs)  
    PRE_AE,RCE_AE = vis.load_ae_result(vs,attrid=[2])
    ae_m = numpy.mean(PRE_AE,axis=2)
    PES = []
    PES.append(PRE_raw)
    ff_id = numpy.argmin(ae_m[:,-1],axis=0)
    gae_id = numpy.argmin(ae_m[:,:])
    gae_id_x,gae_id_y = numpy.unravel_index(gae_id,
                                            ae_m.shape)
    PES.append(PRE_AE[ff_id,-1,:])
    PES.append(PRE_AE[gae_id_x,gae_id_y,:])
    PES = numpy.asarray(PES)
    mstr = ['RAW','NN','GAE']
    plt.figure(figsize=[8,5])
    plt.boxplot(PES.transpose())
    plt.xticks(numpy.arange(PES.shape[0])+1,mstr)
    plt.ylabel('Age Prediction Error')
    plt.show()
    
def fig15_scatter_gae_path():
    font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 15}
    matplotlib.rc('font', **font)
    ms,vs = vis.get_D5_ms_vs()
    PRE_raw,PRE_pca,RCE_pca = vis.load_raw_pca_result(vs)  
    PRE_AE,RCE_AE = vis.load_ae_result(vs,attrid=[0,2])
    w_lst = range(2,11)
    pca_p = numpy.mean(PRE_pca,axis=1)
    pca_r = numpy.mean(RCE_pca,axis=1)
    raw_p = numpy.mean(PRE_raw)
    ae_p,ae_r = numpy.mean(PRE_AE,axis=2),numpy.mean(RCE_AE,axis=2)    
    nc = 9
    cs = get_colors(nc)    
    plt.figure(figsize=[10,4])
    slst = [1,3]
    for i in slst:
        c = cs[i]
        x,y = pca_r[i],pca_p[i]
        plt.plot(x,y,'*',markersize=20,color=c)
#        plt.text(x,y,'PCA-'+str(w_lst[i]))
        plt.xlabel('Cytokine Reconstruction Error')
        plt.ylabel('Age Reconstruction Error')
    plt.plot([0.1,0.8],[raw_p]*2,'k--')
#    plt.text(0.6,raw_p+0.05,'age prediction from features')
    for i in slst:
        c = cs[i]
        plt.plot(ae_r[i,:-1],ae_p[i,:-1],'-o',color=c)
#        plt.text(ae_r[i,-2],ae_p[i,-2],'GAE-'+str(w_lst[i]))    
    plt.xlim([0.1,0.8])
    plt.show()
    
    
#    plt.plot(pca_r,pca_p,'-bo')
#    plt.plot(ae_r[:,0],ae_p[:,0],'-go')
#    plt.plot(gae_r,gae_p,'-ro')
#    plt.plot([0.1,0.5],[raw_p]*2,'k--')
#    plt.plot([0.1,0.5],[nn_p]*2,'c--')
#    plt.grid()


def fig3_Jacobian():
    font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 12}
    matplotlib.rc('font', **font)
    dataopt = DataOpts(test_folds=5,valid_folds=3)
    data = load_data(dataopt)
    ds,dsv = data.get_all()
    x,y,xn = ds
    ms,vs = data.transformation[1]
    
    raw,pca2,nn,gae = loadobj('temp/D5-1-best-models.pkl')
    py = gae.predict(x)
    page = py*vs + ms
    age = y*vs + ms
    rage = numpy.abs(page-age)
    lin_coef = raw.enet.coef_
    J = get_Jacobian(gae,x)
    
    E = get_RCE_residual(gae,x)
    C = gae.encode(x)
    xlb = data.x_labels
    p = len(xlb)
    nms = ['Correlation','LinearCoef','Jacobian','CytoResidual']
    
    # Jacobian and CytoRes
    plt.figure(figsize=[12,6])
    plt.subplot(2,1,1)
    plt.boxplot(J)
    plt.xticks(numpy.arange(p)+1,xlb,rotation='vertical')
    plt.ylabel('Partial Derivatives')
    plt.grid()
    plt.xlim([0,48])
    plt.subplot(2,1,2)
    bart(lin_coef,xlb)
    plt.ylabel('Linear Coefficients')
#    plt.grid()
    plt.xlim([0,48])
    plt.tight_layout()
    plt.show()
#    
#    S_p = numpy.mean(numpy.abs(J),axis=1)
#    plt.scatter(C[:,0],C[:,6],c=S_p,cmap='hot')
#    plt.xlabel('Code 0')
#    plt.ylabel('Code 6')
#    plt.colorbar()
#    plt.xlim([-1.1,1.1])
#    plt.ylim([-1.1,1.1])
#    plt.title('Age Prediction Sensitivity')
#    plt.show()
#    
#    plt.scatter(rage,S_p)
#    plt.xlabel('Age Prediction Residual')
#    plt.ylabel('Age Prediction Sensitivity')
#    plt.show()
#    
##    S_p = numpy.mean(numpy.abs(J),axis=1)
#    plt.scatter(C[:,0],C[:,6],c=age,cmap='hot')
#    plt.xlabel('Code 0')
#    plt.ylabel('Code 6')
#    plt.colorbar()
#    plt.xlim([-1.1,1.1])
#    plt.ylim([-1.1,1.1])
#    plt.title('Age')
#    plt.show()
#    S_f = numpy.mean((J2),axis=0)
#    plst = numpy.argsort(S_p)
#    flst = numpy.argsort(S_f)
#    J_srt = J[plst,:]
#    J_srt = J_srt[:,flst]
#    plt.subplot(1,2,1)
#    plt.imshow(C[plst,:],aspect='auto',interpolation='none',
#               vmax='1',vmin='-1',cmap='PRGn')
#    plt.subplot(1,2,2)
#    plt.imshow(J_srt,aspect='auto',interpolation='none',
#               vmax = 1, vmin = -1, cmap = 'PRGn')
#    plt.show()

#font = {'family' : 'normal',
#        'weight' : 'normal',
#        'size'   : 22}
#matplotlib.rc('font', **font)
#fig1_Raw_and_PCA_pred_error()    
#fig2_gae_is_the_best_prediction()
#fig15_scatter_gae_path()
fig3_Jacobian()
