# -*- coding: utf-8 -*-
"""
Created on Mon Mar 06 20:25:40 2017

@author: cling
"""

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy
from utils import loadobj,get_colors,spearmancorr,corr,bart
from data_gen import DataOpts,load_data,Combine_dataset
from exp_test import calc_best_test_err,combine_ER
from trainer import get_Jacobian, get_RCE_residual
def plot_bar_mean(allAER,titstr,attrid):
        xlbs = []
        n = len(allAER)
        test_folds = allAER[0].best_test_errs.shape[0]
#        number_of_lines = test_folds
#        cm_subsection = numpy.linspace(0, 1, number_of_lines) 
#        colors = [ cm.jet(x) for x in cm_subsection ]
#        T = allAER[0].test_errs.shape[0]
        rel_errs = numpy.zeros([test_folds,n,8])
        for play in range(test_folds):
            for ER,i in zip(allAER,range(n)):
                    xs = ER.best_test_errs[play,:]
#                    xs = allAER[0].best_test_errs[play,:]                
                    rel_errs[play,i,:] = xs
#                    lg = plt.scatter(i+1, xs[attrid], color=c,s=20)
#                    lg = plt.bar(numpy.mean())
                    if play==0:
                        xlbs.append(ER.name)
        E = numpy.mean(rel_errs[:,:,attrid],axis=0)
        plt.bar(range(1,n+1),E)
        plt.title(titstr)
        plt.xticks(range(1,n+1),xlbs,rotation='vertical')
        plt.ylabel('relative-Error')

def load_gae_all_result(vs,attrid=[0,2]):
    aes,pars = loadobj('temp/D5-1-AE.pkl')  
    ae = combine_ER(numpy.concatenate(aes,axis=0),'GAE-ALL')
    ae = calc_best_test_err(ae,attrid)
    RCE = ae.best_test_errs[:,0]
    PRE_AE = numpy.sqrt(ae.best_test_errs[:,2])*vs
    RCE_AE = RCE
    return PRE_AE,RCE_AE

def load_gae_wwise_result(vs,attrid=[0,2]):
    aes,pars = loadobj('temp/D5-1-AE.pkl')  
    w_lst_ae = range(2,11)
    m = len(w_lst_ae)
    PRE_AE = numpy.zeros([m,5])
    RCE_AE = numpy.zeros([m,5])
    for i in range(m):
        ae = combine_ER(aes[i],'GAE-'+str(w_lst_ae[i]))
        print ae.test_errs.shape
        print ae.valid_errs.shape
        print ae.paras.shape
        ae = calc_best_test_err(ae,attrid)
        RCE = ae.best_test_errs[:,0]
        PRE_AE[i,:] = numpy.sqrt(ae.best_test_errs[:,2])*vs
        RCE_AE[i,:] = RCE
    return PRE_AE,RCE_AE
        
def load_ae_result(vs,attrid=[0,2]):
    aes,pars = loadobj('temp/D5-1-AE.pkl')  
    w_lst_ae = range(2,11)
    lam_lst = [0,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,1.0]
    m,n = len(w_lst_ae),len(lam_lst)
    PRE_AE = numpy.zeros([m,n,5])
    RCE_AE = numpy.zeros([m,n,5])
    for i in range(0,len(w_lst_ae)):
        for j in range(len(lam_lst)):
            ae = calc_best_test_err(aes[i][j],attrid)
            RCE = ae.best_test_errs[:,0]
            PRE_AE[i,j,:] = numpy.sqrt(ae.best_test_errs[:,2])*vs
            RCE_AE[i,j,:] = RCE
    return PRE_AE,RCE_AE

def load_raw_pca_result(vs):
    raw,pca = loadobj('temp/D5-1-rawpca.pkl')  
    PRE_raw = numpy.sqrt(raw.best_test_errs[:,2])*vs
    PRE_pca = numpy.sqrt(pca.test_errs[1:10,:,2])*vs
    RCE_pca = pca.test_errs[1:10,:,0]
    return PRE_raw,PRE_pca,RCE_pca
    
def get_test_err(name='AE',attrid=[0,2]):
    raw,pca = loadobj('temp/D5-1-rawpca.pkl')  
    mus = []
    vs = []
    for i in range(5):
        data = load_data(dataopt=DataOpts(test_folds=5,valid_folds=3,test_fold_id=i))
        data.get_test()
        m,v = data.transformation[1]
        mus.append(m[0])
        vs.append(v[0])
#        print m,v
    vs = numpy.asarray(vs)
#    print vs.shape
    PRE_raw = numpy.sqrt(raw.best_test_errs[:,2])*vs
    PRE_pca = numpy.sqrt(pca.test_errs[:,:,2])*vs
    RCE_pca = pca.test_errs[:,:,0]
    PRE_AE,PRE_AE_mean,RCE_AE,RCE_AE_mean,w_lst_ae = load_ae_result(vs,attrid)
    if name=='RAW':
        return PRE_raw
    elif name=='PCA':
        return RCE_pca[1:10,:],PRE_pca[1:10,:]
    elif name=='AE':
        return RCE_AE[:,0,:],PRE_AE[:,0,:]
    elif name=='GAE':
        bre = numpy.zeros(RCE_AE[:,0,:].shape)
        bpe = numpy.zeros(RCE_AE[:,0,:].shape)
        for i in range(RCE_AE.shape[0]):
            gae_id = numpy.argmin(PRE_AE_mean[i,:])
            bre[i,:] = RCE_AE[i,gae_id,:]
            bpe[i,:] = PRE_AE[i,gae_id,:]
        return bre,bpe

def get_D5_ms_vs():
    mus = []
    vs = []
    for i in range(5):
        data = load_data(dataopt=DataOpts(test_folds=5,valid_folds=3,test_fold_id=i))
        data.get_test()
        m,v = data.transformation[1]
        mus.append(m[0])
        vs.append(v[0])
    vs = numpy.asarray(vs)
    return mus,vs

def show_top_influence_on_pre():
    aes,ps = loadobj('temp/D5-1-TOP-AE.pkl')
    mus,vs = get_D5_ms_vs()
    top_lst = [0,5,10,20,30,40,50]
#    lam_lst = [0,0.1,0.2,0.3,0.4,0.5,1.0]    
    n = len(top_lst)
    PRE_AE = numpy.zeros([n,5])
    st = []
    for i in range(n):
        ae = calc_best_test_err(aes[i],[6])
        PRE_AE[i,:] = numpy.sqrt(ae.best_test_errs[:,2])*vs
        print ae.best_settings
        st.append(str(top_lst[i]))
    plt.plot(range(1,8),numpy.mean(PRE_AE,axis=1),'-o')
    plt.boxplot(PRE_AE.transpose())
    plt.grid()
    st[-1] = '47'
    plt.xticks(range(1,8),st)
    plt.xlabel('Number of Cytokine Reconstruction Included (total 47)')
    plt.ylabel('Age prediction Error')

def visualize():
    plt.figure(figsize=[10,10])
    raw,pca = loadobj('temp/D5-1-rawpca.pkl')  
    mus = []
    vs = []
    for i in range(5):
        data = load_data(dataopt=DataOpts(test_folds=5,valid_folds=3,test_fold_id=i))
        data.get_test()
        m,v = data.transformation[1]
        mus.append(m[0])
        vs.append(v[0])
#        print m,v
    vs = numpy.asarray(vs)
#    print vs.shape
#    nc = pca.test_errs.shape[0]
    w_lst = range(1,11)
    nc = len(w_lst)
    PRE_raw = numpy.sqrt(raw.best_test_errs[:,2])*vs
    PRE_pca = numpy.sqrt(pca.test_errs[:,:,2])*vs
    RCE_pca = pca.test_errs[:,:,0]

    PRE_raw_mean = numpy.mean(PRE_raw)
    PRE_pca_mean = numpy.mean(PRE_pca[:,:],axis=1)
    RCE_pca_mean = numpy.mean(RCE_pca,axis=1)
        
    cs = get_colors(nc)    
    for i in range(nc):
        c = cs[i]
        x,y = RCE_pca_mean[i],PRE_pca_mean[i]
        plt.plot(x,y,'x',color=c)
        plt.text(x,y,'PCA-'+str(w_lst[i]))
        plt.xlabel('RCE')
        plt.ylabel('PRE')
    plt.plot([0,0.9],[PRE_raw_mean,PRE_raw_mean],'k--')
    
    PRE_AE,PRE_AE_mean,RCE_AE,RCE_AE_mean,w_lst_ae = load_ae_result(vs,attrid=[0,2])
    for i in [1,2]:
        c = cs[i]
        plt.plot(RCE_AE_mean[i,:-1],PRE_AE_mean[i,:-1],'-o',color=c)
        plt.text(RCE_AE_mean[i,-2],PRE_AE_mean[i,-2],'GAE-'+str(w_lst_ae[i]))    
    plt.show()
    
    # Reconstruction Error
    PRE_AE,PRE_AE_mean,RCE_AE,RCE_AE_mean,w_lst_ae = load_ae_result(vs,attrid=[0])
    w2_lst = range(2,11)
    pca_v = numpy.std(RCE_pca[1:10,:],axis=1)/5
    ae_v = numpy.std(RCE_AE[:,0,:],axis=1)/5
    plt.errorbar(w2_lst,RCE_pca_mean[1:10],yerr=pca_v,label='PCA')
    plt.errorbar(w2_lst,RCE_AE_mean[:,0],yerr=ae_v,label='AE')
    plt.xlim([1,12])
    plt.legend()
    plt.xlabel('Code Length')
    plt.ylabel('Cytokine Reconstruction Error')
    plt.show()    
    
    # AE and PCA code is not predictive of the age
    PRE_AE,PRE_AE_mean,RCE_AE,RCE_AE_mean,w_lst_ae = load_ae_result(vs,attrid=[2])
#    RCE_GAE,PRE_GAE = get_test_err(name='GAE',attrid=[2])
#    PRE_GAE_mean = numpy.mean(PRE_GAE,axis=1)
    w2_lst = range(2,11)
    pca_v = numpy.std(PRE_pca[1:10,:],axis=1)/5
    ae_v = numpy.std(PRE_AE[:,0,:],axis=1)/5
#    gae_v = numpy.std(PRE_GAE,axis=1)/5
    plt.errorbar(w2_lst,PRE_pca_mean[1:10],yerr=pca_v,label='PCA')
    print PRE_AE_mean.shape
    plt.errorbar(w2_lst,PRE_AE_mean[:,0],yerr=ae_v,label='AE')
#    plt.errorbar(w2_lst,PRE_GAE_mean,yerr=gae_v,label='GAE')
    plt.plot([1,12],[PRE_raw_mean]*2,'k--',label='RAW')
    plt.xlim([1,12])
    plt.legend()
    plt.xlabel('Code Length')
    plt.ylabel('Age Prediction Error')
    plt.show()
    
    PES = []
    PES.append(PRE_raw)
    pca_id = numpy.argmin(PRE_pca_mean[:11],axis=0)
    print pca_id
    PES.append(PRE_pca[pca_id,:])
    ae_id = numpy.argmin(PRE_AE_mean[:,0],axis=0)
    ff_id = numpy.argmin(PRE_AE_mean[:,-1],axis=0)
    gae_id = numpy.argmin(PRE_AE_mean[:,:])
    gae_id_x,gae_id_y = numpy.unravel_index(gae_id,PRE_AE_mean.shape)
    PES.append(PRE_AE[ae_id,0,:])
    PES.append(PRE_AE[ff_id,-1,:])
    PES.append(PRE_AE[gae_id_x,gae_id_y,:])
    PES = numpy.asarray(PES)
    mstr = ['RAW','PCA','AE','NN','GAE']
    plt.boxplot(PES.transpose())
    plt.xticks(numpy.arange(PES.shape[0])+1,mstr)
    plt.ylabel('PRE')
    plt.show()

def boxplot_ageres(gender,R,cnames,xlabel):
    lst = numpy.where(~numpy.isnan(gender))[0]
    gender = gender[lst]
    R = R[lst]
    l0 = numpy.where(gender==0)[0]
    l1 = numpy.where(gender==1)[0]
    ds = [R[l0],R[l1]]
    plt.boxplot(ds)
    plt.xticks([1,2],cnames)
    plt.xlabel(xlabel)
    plt.ylabel('Absolute Age Prediction Error')
    
def scatter_ageres(gender,R,xlabel):
    lst = numpy.where(~numpy.isnan(gender))[0]
    gender = gender[lst]
    R = R[lst]
    plt.scatter(gender,R)
    plt.xlabel(xlabel)
    plt.ylabel('Absolute Age Prediction Error')
    

def distribution_age_residual():
    dataopt = DataOpts(test_folds=5,valid_folds=3)
    D = Combine_dataset(dataopt)
    gender = D.gender
    data = load_data(dataopt)
    ds,dsv = data.get_all()
    x,y,xn = ds
    m,v = data.transformation[1]
    raw,pca2,nn,gae = loadobj('temp/D5-1-best-models.pkl')
    py = gae.predict(x)
    age = m+y*v
    page = m+py*v
    
    plt.scatter(m+y*v,m+py*v)
    plt.plot([0,100],[0,100],'k--')
    plt.xlim([0,100])
    plt.ylim([0,100])
    plt.xlabel('True Age')
    plt.ylabel('GAE Predicted Age')
    plt.show()
    
    R = numpy.abs(page-age)
    plt.hist(R)
    plt.xlabel('Absolute Age Prediction Error')
    plt.ylabel('Count')
    plt.show()

#    plt.scatter(age,R)
#    plt.xlabel('True Age')
#    plt.ylabel('Absolute Age Prediction Error')
#    plt.show()
    scatter_ageres(age,R,'True Age'),plt.show()

    xs = [gender,D.source,D.cmv,D.ebv]
    clsnm = [['Female','Male'],['Montoya','Davis'],['-','+'],['-','+']]
    xb = ['Gender','Source','CMV','EBV']
    plt.figure(figsize=[8,8])
    for i in range(4):
        plt.subplot(2,2,i+1)
        boxplot_ageres(xs[i],R,clsnm[i],xb[i])
    plt.tight_layout()
    plt.show()
    
    xs = [age,D.bmi,D.mfs]
    xb = ['Age','BMI','MFS'] 
    plt.figure(figsize=[12,4])
    for i in range(3):
        plt.subplot(1,3,i+1)
        scatter_ageres(xs[i],R,xb[i])
    plt.tight_layout()
    plt.show()
#    l0 = numpy.where(gender==0)[0]
#    l1 = numpy.where(gender==1)[0]
#    ds = [R[l0],R[l1]]
#    plt.boxplot(ds)
#    plt.xticks([1,2],)
#    plt.xlabel('Gender')
#    plt.ylabel('Absolute Age Prediction Error')
#    plt.show()

    
#    l0 = numpy.where(gender==0)[0]
#    l1 = numpy.where(gender==1)[0]
#    ds = [R[l0],R[l1]]
#    plt.boxplot(ds)
#    plt.xticks([1,2],['Female','Male'])
#    plt.xlabel('Gender')
#    plt.ylabel('Absolute Age Prediction Error')
#    plt.show()
    

def plot_code_vs_age(code,age):
#    c = get_colors(100)
    age = numpy.round(age).astype('int')
    clrmap = plt.cm.get_cmap('viridis')
#    print clrmap.shape
    plt.scatter(code[:,0],code[:,1],c=age[:,0],s=10,cmap=clrmap,
                edgecolors='none')
    plt.colorbar()


def show_code_plot(pca2,gae,x,age,tit):
    lst = numpy.where(~numpy.isnan(age[:,0]))[0]
    age = age[lst,:]
    x = x[lst,:]    
    
    plt.figure(figsize=[8,4])
    plt.subplot(1,2,1)
    plot_code_vs_age(pca2.encode(x),age)
    plt.xlabel('PCA-1')
    plt.ylabel('PCA-2')
    plt.title(tit)
    plt.subplot(1,2,2)
    code = gae.encode(x)
    code = code[:,[0,-1]]
    plot_code_vs_age(code,age)
    plt.xlabel('GAE Code 0')
    plt.ylabel('GAE Code 6')
    plt.tight_layout()
    plt.title(tit)    
    plt.xlim([-1.1,1.1])
    plt.ylim([-1.1,1.1])
    plt.show()
    

def show_code_and_age():
    dataopt = DataOpts(test_folds=5,valid_folds=3)
    C = Combine_dataset(dataopt)
    data = load_data(dataopt)
    ds,dsv = data.get_all()
    x,y,xn = ds
    ms,vs = data.transformation[1]
    age = y*vs + ms
    raw,pca2,nn,gae = loadobj('temp/D5-1-best-models.pkl')
    tos = [age,C.source,C.gender,C.bmi,C.mfs,C.cmv,C.ebv]
    tits = ['Age','Source','Gender','BMI','MFS','CMV','EBV']
    for i in range(7):
        show_code_plot(pca2,gae,x,tos[i],tits[i])
#    show_code_plot(pca2,gae,x,)

def show_general_correlations():
    dataopt = DataOpts(test_folds=5,valid_folds=3)
    data = load_data(dataopt)
    ds,dsv = data.get_all()
    x,y,xn = ds
    ms,vs = data.transformation[1]
    age = y*vs + ms
    cyto_scorr = spearmancorr(y,x)
    raw,pca2,nn,gae = loadobj('temp/D5-1-best-models.pkl')
    lin_coef = raw.enet.coef_
    J = get_Jacobian(gae,x)
    E = get_RCE_residual(gae,x)
    gae_J = numpy.mean(J,axis=0)
    gae_E = numpy.mean(E,axis=0)
    rs = [cyto_scorr,lin_coef,gae_J,gae_E]
    xlb = data.x_labels
    p = len(xlb)
    nms = ['Correlation','LinearCoef','Jacobian','CytoResidual']
   
    # Raw correlation and Linear Coefs
    plt.figure(figsize=[10,6])    
    for i in range(2):
        plt.subplot(2,1,i+1)
        bart(rs[i].transpose(),xlb)
        plt.ylabel(nms[i])
        plt.xlim([0,48])
    plt.tight_layout()
    plt.show()
        
   
    # Jacobian and CytoRes
    plt.figure(figsize=[10,6])
    plt.subplot(2,1,1),plt.boxplot(J)
    plt.xticks(numpy.arange(p)+1,xlb,rotation='vertical')
    plt.ylabel('Jacobian')
    plt.grid()
    plt.xlim([0,48])
    plt.subplot(2,1,2),plt.boxplot(E)
    plt.xticks(numpy.arange(p)+1,xlb,rotation='vertical')
    plt.ylabel('Cytokine Reconstruction Residual')
    plt.grid()
    plt.xlim([0,48])
    plt.tight_layout()
    plt.show()
    
    # correlation between age measures
    plt.figure(figsize=[10,10])
    c = 0
    for i in range(4):
        for j in range(4):
            c = c + 1
            plt.subplot(4,4,c)
            plt.scatter(rs[i],rs[j])
            plt.xlabel(nms[i])
            plt.ylabel(nms[j])
    plt.tight_layout()    
    plt.show()
    
    
    # Variance in Jacobian
    plt.figure(figsize=[10,4])
    Vj = numpy.std(J,axis=0)
    lst = numpy.argsort(-Vj)
    TOP = 10
    tlst = lst[:TOP]
    plt.boxplot(J[:,tlst])
    plt.xticks(numpy.arange(TOP)+1,xlb[tlst])
    plt.ylabel('Jacobian')
    plt.xlabel('Top 10 most variable Jacobian')
    plt.grid()
    plt.show()
    
    # code's correlation with Age
    C = gae.encode(x)
    C_corr = corr(C,y)
    plt.figure(figsize=[12,6])
    for i in range(7):
        plt.subplot(2,4,i+1)
        plt.scatter(C[:,i],age,s=5)
        plt.xlabel('Code ' + str(i))
        plt.ylabel('True Age')
    plt.subplot(2,4,8)
    bart(C_corr)
    plt.xlabel('Code')
    plt.ylabel('Corr with Age')
    plt.tight_layout()
    plt.show()

def analyze_Jacobian():
    dataopt = DataOpts(test_folds=5,valid_folds=3)
    data = load_data(dataopt)
    ds,dsv = data.get_all()
    x,y,xn = ds
    ms,vs = data.transformation[1]
    age = y*vs + ms
    raw,pca2,nn,gae = loadobj('temp/D5-1-best-models.pkl')
    J = get_Jacobian(gae,x)
    C = gae.encode(x)
    plt.figure(figsize=[10,5])
    plt.subplot(1,2,1)
    plt.imshow(C,aspect='auto',interpolation='none',cmap='PRGn')
    plt.subplot(1,2,2)    
    plt.imshow(J,aspect='auto',interpolation='none',cmap='PRGn',vmax=1.5,vmin=-1.5)    
    plt.colorbar()
    plt.show()
    
def visual_ND5():
    raw = loadobj('temp/ND5-raw.pkl')
    aes,tp = loadobj('temp/ND5-AE.pkl')
    errs = [raw.best_test_errs[:,2]]
    numpy.set_printoptions(precision=2,suppress=True)
    for ae in aes[:-1]:
        ae = calc_best_test_err(ae,[2])
        errs.append(ae.best_test_errs[:,2])
        print ae.best_settings
    errs = numpy.asarray(errs)
    plt.boxplot(errs.transpose())
#    plt.plot(errs.transpose())
    print errs
    plt.xticks(range(1,5),['RAW','NN','GAE0.1','GAE0.5'])
if __name__ == '__main__':
    visual_ND5()
#    plt.figure(figsize=[])

#        print r.shape
#        bart(r.transpose(),)
#        plt.show()
#distribution_age_residual()
#visualize()
#print get_test_err(name='GAE',attrid=[2])
#show_general_correlations()
#show_code_and_age()
#show_top_influence_on_pre()
#analyze_Jacobian()