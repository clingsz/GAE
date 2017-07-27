# -*- coding: utf-8 -*-
"""
Created on Wed Feb 01 14:45:45 2017

@author: cling
"""

# look at non-linear codes
from misc.cv.exp_test import makeTest,ExpResult
from misc.utils import saveobj,getJobOpts,loadobj
import numpy
from misc.data_gen import DataOpts,load_data
from gae.model.trainer import TrainerOpts,make_trainer
import cPickle
from gae.model.guidedae import GAEOpts

def get_raw_result(dataopt):
    T = dataopt.test_folds
    bl_res = []    
    for did in range(T):    
        dataopt.test_fold_id=did
        t = makeTest(dataopt=dataopt,traineropt=TrainerOpts(name='None'))
        errs = t.test()
        bl_res.append(errs)
    raw_test_errs = numpy.asarray(bl_res)
    blER = ExpResult('RAW')
    blER.load_test_errs(raw_test_errs)
    return blER
    
def get_pca_result(dataopt,lst=range(1,30)):
    T = dataopt.test_folds
    valid_folds = dataopt.valid_folds
    ncs = lst
    n = len(ncs)
    verrs = numpy.zeros([n,T,valid_folds,8])
    terrs = numpy.zeros([n,T,8])
    for did in range(T):    
        dataopt.test_fold_id=did
        for i in range(len(ncs)):
            nc = ncs[i]
            t = makeTest(dataopt=dataopt,traineropt=TrainerOpts(name='PCA',n_components=nc))
            errs = t.test()
            terrs[i,did,:] = errs
    ncs = numpy.asarray(ncs).reshape([n,1])
    pcaER = ExpResult('PCA')
    pcaER.load_exp_result(verrs,terrs,ncs)
    return pcaER


def get_exp_result(ExpName,dataopt):
    blER = get_raw_result(dataopt)
    pcaER = get_pca_result(dataopt)
    saveobj('temp/' + ExpName + '-rawpca.pkl',[blER,pcaER])

def collect_ND5_result():
    file_name = 'result/logs_ND5.pkl'
    res = cPickle.load(open(file_name,'rb'))
    
    w_lst = [2,5,10,15,20]
    d_lst = [1,2,3]     
    lam_lst = [0,0.1,0.5,1.0] 
    l2_lst = [1e-3,1e-2,1e-1]
    ns_lst = [0,0.1,0.2,0.3,0.4]    
    joblsts = [w_lst,d_lst,lam_lst,l2_lst,ns_lst]
#    w,d,lam,l2,ns = paras
    test_folds = 5
    valid_folds = 3
    trials = len(res)
    na = 8
    test_errs = numpy.zeros([trials,test_folds,na])
    valid_errs = numpy.zeros([trials,test_folds,valid_folds,na])
    paras = numpy.empty([trials,len(joblsts)])
    c = 0
    for j in range(trials):
        rt = res[c]
        paras[j,:] = getJobOpts(c,joblsts)
        for i in range(test_folds):
            r = rt[i+1]
            for k in range(valid_folds):
                valid_errs[j,i,k,:] = r[k]
            test_errs[j,i,:] = r[-1]
        c = c + 1   
    
    aes = []
    for lam in lam_lst:
        er = ExpResult('GAE'+str(lam))
        lim = numpy.where(paras[:,0]<15)[0]
        lst = numpy.where(paras[:,2]==lam)[0]
        lst = numpy.intersect1d(lim,lst)
        er.load_exp_result(valid_errs[lst,:,:,:],test_errs[lst,:,:],paras[lst,:])
        aes.append(er)
    saveobj('temp/ND5-AE.pkl',[aes,(test_errs,paras)])

def collect_GAE_result():
    file_name = 'result/logs_D5_1.pkl'
    res = cPickle.load(open(file_name,'rb'))
    did_lst = range(5)
    w_lst = range(2,11)
    d_lst = [1,2,3]
    wr_lst = [1,2]    
    lam_lst = [0,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,1.0]    
    joblsts = [did_lst,w_lst,d_lst,wr_lst,lam_lst]    
    test_folds = 5
    valid_folds = 3
    trials = len(res)/test_folds
    na = 8
    test_errs = numpy.zeros([trials,test_folds,na])
    valid_errs = numpy.zeros([trials,test_folds,valid_folds,na])
    paras = numpy.empty([trials,len(joblsts)-1])
    c = 0
    for i in range(len(did_lst)):
        for j in range(trials):
            r = res[c]
            paras[j,:] = getJobOpts(c,joblsts)[1:]
            for k in range(valid_folds):
                valid_errs[j,i,k,:] = r[k+1]
            test_errs[j,i,:] = r[-1]
            c = c + 1   
    
    aes = []
    for w in w_lst:
        ael = []
        for lam in lam_lst:
            er = ExpResult('GAE'+str(lam)+'-w'+str(w))
            l1 = numpy.where(paras[:,-1]==lam)[0]
            l2 = numpy.where(paras[:,0]==w)[0]
            lst = numpy.intersect1d(l1,l2)
            er.load_exp_result(valid_errs[lst,:,:,:],test_errs[lst,:,:],paras[lst,:])
            if lam==1:
                er.test_errs[:,:,0] = 1
                er.valid_errs[:,:,:,0] = 1
            ael.append(er)
        aes.append(ael)
    saveobj('temp/D5-1-AE.pkl',[aes,(test_errs,paras)])

def collect_GAETOP_result():
    file_name = 'result/logs_D5_1_TOP.pkl'
    res = cPickle.load(open(file_name,'rb'))
    did_lst = range(5)
    w_lst = range(2,11)
    d_lst = [1,2]     
    top_lst = [0,5,10,20,30,40,50]
    lam_lst = [0,0.1,0.2,0.3,0.4,0.5,1.0]    
    joblsts = [did_lst,w_lst,d_lst,top_lst,lam_lst]
    test_folds = 5
    valid_folds = 3
    trials = len(res)/test_folds
    na = 8
    test_errs = numpy.zeros([trials,test_folds,na])
    valid_errs = numpy.zeros([trials,test_folds,valid_folds,na])
    paras = numpy.empty([trials,len(joblsts)-1])
    c = 0
    for i in range(len(did_lst)):
        for j in range(trials):
            r = res[c]
            paras[j,:] = getJobOpts(c,joblsts)[1:]
            for k in range(valid_folds):
                valid_errs[j,i,k,:] = r[k+1]
            test_errs[j,i,:] = r[-1]
            c = c + 1   
    
    aes = []
    for t in top_lst:
        er = ExpResult('GAE-t'+str(t))
        l1 = numpy.where(paras[:,-2]==t)[0]
#        l3 = numpy.where(paras[:,-4]==7)[0]
        l2 = numpy.where(paras[:,-1]>0)[0]
#        l4 = numpy.where(paras[:,-3]==2)[0]
        lst = numpy.intersect1d(l1,l2)
#        lst = numpy.intersect1d(lst,l3)
#        lst = numpy.intersect1d(lst,l4)
        er.load_exp_result(valid_errs[lst,:,:,:],test_errs[lst,:,:],paras[lst,:])
        aes.append(er)
    saveobj('temp/D5-1-TOP-AE.pkl',[aes,(test_errs,paras)])

def fetch_best_params(model=2,attrid=[2]):
    aes,tp = loadobj('temp/D5-1-AE.pkl')
    terrs,paras = tp
#    print terrs.shape
    lst = None
    if attrid==0:
        lst = numpy.where(paras[:,-1]<1)[0]
        terrs = terrs[lst,:,:]
        paras = paras[lst,:]
    lst = None
    if model==0:
        lst = numpy.where(paras[:,-1]==0)[0]
    if model==1:
        lst = numpy.where(paras[:,-1]==1)[0]
    if lst is not None:
        terrs = terrs[lst,:,:]
        paras = paras[lst,:]
    if len(attrid)>0:
        s_terrs = numpy.sum(terrs[:,:,attrid],axis=2)
    else:
        s_terrs = terrs[:,:,attrid]
#    print terrs.shape
    
    bid = numpy.argmin(numpy.mean(s_terrs,axis=1))
    print paras[bid,:]
    print numpy.mean(terrs[bid,:,:],axis=0)
    return terrs[bid,:,attrid]
    
def run_best_model_on_all():
    dataopt = DataOpts(test_folds=5,valid_folds=3)
    data = load_data(dataopt)
    ds = data.get_all()
    raw = make_trainer(traineropt = TrainerOpts(name='None'))
    raw.train(ds)    
    pca2 = make_trainer(traineropt = TrainerOpts(name='PCA',n_components=2))
    pca2.train(ds)
#    aeopt = AEOpts(w=10,wr=2,d=2,noise_level=0,
#                   lam=0,decoder_l2r=0,verbose=1)
    nnopt = GAEOpts(w=5,wr=1,d=2,noise_level=0,
                   lam=1,decoder_l2r=0,verbose=1)
    gaeopt = GAEOpts(w=7,wr=1,d=2,lam=0.3,noise_level=0,
                    decoder_l2r=0,verbose=1)
    nn = make_trainer(traineropt = TrainerOpts(name='AE',aeopt=nnopt))
    gae = make_trainer(traineropt = TrainerOpts(name='AE',aeopt=gaeopt))
    nn.train(ds)
    gae.train(ds)
    saveobj('temp/D5-1-best-models.pkl',(raw,pca2,nn,gae))    
    
#    ae = make_trainer()

def run():
    dataopt = DataOpts(test_folds=5,valid_folds=3)
    data = load_data(dataopt)
    ds = data.get_test()
    aeopt = GAEOpts(w=5,d=2,wr=1,noise_level=0,lam=0.3,l2=1e-2,verbose=1)
    raw = make_trainer(traineropt = TrainerOpts(name='None'))
    pca = make_trainer(traineropt = TrainerOpts(name='PCA',n_components=5))
    gae = make_trainer(traineropt = TrainerOpts(name='AE',aeopt=aeopt))
    raw.train_and_test(ds)
    pca.train_and_test(ds)
    gae.train_and_test(ds)

if __name__ == '__main__':
     run() 