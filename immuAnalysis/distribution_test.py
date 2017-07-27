# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 21:43:24 2017

@author: cling
"""

from scipy.stats import norm,lognorm,laplace,loglaplace,gamma,loggamma
import misc.data_gen as dg
import numpy
import misc.utils as utils
from sklearn.model_selection import KFold
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib

def get_data():
    data = dg.get_processed_data()
    demo = data['demo']
    D = []
    D.append(demo['age'])
    D.append(demo['bmi'])
    D.append(demo['mfs'])
    D.append(data['cyto'])
    D.append(data['flow'])
    nms = ['AGE','BMI','MFS'] + data['cyto_names'].tolist() + data['flow_names'].tolist()
    Ds = numpy.concatenate(D,axis=1)
    return Ds,nms    

def distribution_test_run():
    Ds,nms = get_data()
    ll_cvs = []
    lls = []
#    rng = numpy.random.RandomState(0)
    rng = 0
    p = Ds.shape[1]
    for i in range(p):
        x = Ds[:,i]
        ll_cv,ll_all = dist_fit_cv(x,rng)
        ll_cvs.append(ll_cv)
        lls.append(ll_all)
        print '%s, %.2f' % (nms[i],i*100.0/p)
        print numpy.mean(numpy.asarray(ll_cv),axis=0)
    utils.saveobj('result/temp/dist_test_log.pkl',[ll_cvs,lls,nms])
    
def dist_fit_cv(x,rand_state):
    lst = numpy.where(~numpy.isnan(x))[0]
    x = x[lst]
    folds = 5
    kf = KFold(n_splits=folds, shuffle=True, random_state=rand_state)
    kf.get_n_splits(x)
    ll_cv = []
    for train,valid in kf.split(x):
        ll = distribution_fit(x[train],x[valid])
        ll_cv.append(ll)
    ll_all = distribution_fit(x,x)
    return ll_cv,ll_all
        
def get_dist_names():
    dname = ['norm','laplace','lognorm','loglaplace','gamma','loggamma']
    return dname    

def distribution_fit(x,xt=None,verbose=False):
    if xt is None:
        xt = x
    dist = [norm,laplace,lognorm,loglaplace,gamma,loggamma]    
    dname = get_dist_names()
    lls = []
    for nm,d,i in zip(dname,dist,range(len(dist))):
        if i<2:            
            pars = d.fit(x)
            ll = d.logpdf(xt,loc=pars[0],scale=pars[1])
        else:
            x = x + 1e-3
            pars = d.fit(x)            
            ll = d.logpdf(xt,pars[0],loc=pars[1],scale=pars[2])
        if verbose:
            print nm,numpy.mean(ll),pars
        lls.append(numpy.mean(ll))
    return lls

def get_valid_num(x):
    lst = numpy.where(~numpy.isnan(x))[0]
    return x[lst]

def fit_dist(x,x_show,dist_id):
    dist = [norm,laplace,lognorm,loglaplace,gamma,loggamma]
    d = dist[dist_id]
    pars = d.fit(x)                    
    if len(pars)==2:
        ll = d.pdf(x_show,loc=pars[0],scale=pars[1])
    else:
        ll = d.pdf(x_show,pars[0],loc=pars[1],scale=pars[2])
    return ll
    
def show_hist(lst=None,filename='dist_all.pdf'):
    font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 15}
    matplotlib.rc('font', **font)
    
    Ds,nms = get_data()
    if lst is None:
        p = Ds.shape[1]   
        lst = range(p)
    else:
        p = len(lst)
    col = 2
    row = int(p//col)+1    
    plt.figure(figsize=[col*8,row*4])
    bd,bpv = utils.loadobj('result/temp/best_dist_ids.pkl')
    dnms = get_dist_names()
    for j in range(p):
        plt.subplot(row,col,j+1)
        i = lst[j]
        x = get_valid_num(Ds[:,i])        
        pt = plt.hist(x,normed=True,bins=20)
        xs = pt[1]
        xs = numpy.linspace(min(xs),max(xs),100)
        if bd[i]>0:        
            ll = fit_dist(x,xs,bd[i])
            plt.plot(xs,ll)            
        ll = fit_dist(x,xs,0)
        plt.plot(xs,ll,'r')   
#        print bpv[i]
        if bd[i]>0:
            st = '*'
            if bpv[i]<0.00005:
                st = '****'
            if bpv[i]<0.0005:
                st = '***'
            elif bpv[i]<0.005:
                st = '**'
            plt.title('%s (%s%s)' % (nms[i],dnms[bd[i]],st))                         
        else:
            plt.title('%s (%s)' % (nms[i],dnms[bd[i]]))
        plt.ylabel('PDF')
    plt.tight_layout()
#    plt.show()
    plt.savefig('result/fig/'+filename,
                    bbox_inches='tight')
    

def distribution_summarize():
    ll_cvs,lls,nms = utils.loadobj('result/temp/dist_test_log.pkl')
    llcv = numpy.asarray(ll_cvs)
    llcv_m = numpy.mean(llcv,axis=1)
    llcv_s = numpy.std(llcv,axis=1)
    pvs = numpy.ones(llcv_m.shape)
    pvs[:,0] = 0.05
    for i in range(llcv.shape[0]):
        cv0 = llcv[i,:,0]
        for j in range(llcv.shape[2]):
            cv2 = llcv[i,:,j]
            if llcv_m[i,j]>llcv_m[i,0]:
                z,pv = stats.ttest_rel(cv0,cv2)
                pvs[i,j] = pv     
    
    fname = 'result/tables/dist_test_log.csv'
    rn = nms
    cn = get_dist_names()
    best_dist_ids = numpy.argmin(pvs,axis=1)
    best_dist = numpy.asarray(cn)[best_dist_ids]
    n = best_dist.shape[0]
    best_dist = best_dist.reshape([n,1])
    x = numpy.concatenate([pvs,best_dist],axis=1)    
    cn = cn + ['Best']
#    print pvs
    best_pv = numpy.min(pvs,axis=1)
    utils.saveobj('result/temp/best_dist_ids.pkl',(best_dist_ids,best_pv))
    utils.save_csv_table(fname,x,cn,rn)



if __name__ == '__main__':
#    distribution_test_run()
#    distribution_summarize()
    show_hist()