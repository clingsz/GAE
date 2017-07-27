# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 11:06:30 2017

@author: cling
"""


import numpy
from misc.data_gen import load_data,DataOpts
from gae.model.trainer import make_trainer,TrainerOpts
from misc.utils import getJobOpts,get_total,saveobj
from gae.model.guidedae import GAEOpts
          
class Test():
    def __init__(self,dataset,trainer):
        self.dataset = dataset
        self.trainer = trainer

    def train_and_test(self,train_data,test_data):
        x,y,nx = train_data
        vx,vy,vnx = test_data
        self.trainer.train(x,y)
        err = self.trainer.test(vx,vy,vnx)
        return err
        
    def validate_fold(self,fold):
        print "%s fold %d use %s" % (self.dataset.name,fold,self.trainer.name)
        train_data,test_data = self.dataset.get_validation_partition(fold=fold)
        return self.train_and_test(train_data,test_data)
    
    def test(self):
        print "%s testing use %s" % (self.dataset.name,self.trainer.name)
        train_data,test_data = self.dataset.get_test()
        return self.train_and_test(train_data,test_data)
    def train_all(self):
        train_data = self.dataset.get_all()
        return self.train_and_test(train_data,train_data)

def calc_best_test_err(er,aw):
    K = er.test_errs.shape[-1]
    er.best_test_errs = numpy.zeros([er.T,K])
    er.best_settings = numpy.zeros([er.T,er.paras.shape[1]])
    for t in range(er.T):       
        C = numpy.squeeze(er.valid_errs[:,t,:,[0,2]])
        C = C[0,:,:] * aw[0] + C[1,:,:] * aw[1]
        mid = numpy.argmin(numpy.max(C,axis=1))
        er.best_test_errs[t,:] = er.test_errs[mid,t,:]
#            print t,mid
        er.best_settings[t,:] = er.paras[mid,:]
    return er
def combine_ER(ERlst,name):
    verrs = []
    terrs = []
    paras = []
    for e in ERlst:
        terrs.append(e.test_errs)
        verrs.append(e.valid_errs)
        paras.append(e.paras)
    verrs = numpy.concatenate(verrs,axis=0)
    terrs = numpy.concatenate(terrs,axis=0)
    paras = numpy.concatenate(paras,axis=0)
    er =   ExpResult(name)  
    er.load_exp_result(verrs,terrs,paras)
    return er
    
class ExpResult(object):
    # input valid_errs should have settings,testfolds,validfolds
    def __init__(self,name):
        self.name = name
        self.best_settings = []
    def load_test_errs(self,best_test_errs):
        T = best_test_errs.shape[0]
        self.best_test_errs = best_test_errs
        self.T = T
    def load_exp_result(self,verrs,terrs,paras):
        T = terrs.shape[1]
        self.T = T
        self.test_errs = terrs
        self.valid_errs = verrs
        self.paras = paras

def make_simple_exp(name,terr1,terr2):
    e = ExpResult(name)
    for i in range(terr1.shape[0]):
        e.test_errs[i,0] = terr1[i]
        e.test_errs[i,1] = terr2[i]
    return e
        
def makeTest(traineropt=None,dataopt=None):
    if traineropt is None:
        traineropt = TrainerOpts()
    dataset = load_data(dataopt)
    model = make_trainer(traineropt)
    test = Test(dataset,model)
    return test

def setTest(jobid=0):
#    did_lst = range(5)
    w_lst = range(1,11)
    d_lst = [1,2,3]     
    lam_lst = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0] 
    l2_lst = [1e-3,1e-2,1e-1]
    joblsts = [w_lst,d_lst,lam_lst,l2_lst]
    get_total(joblsts)
    paras = getJobOpts(jobid,joblsts)
    w,d,lam,l2 = paras
    print paras
    aeopt = GAEOpts(w=w,wr=1,d=d,
                   lam=lam,verbose=1,
                   l2=l2,batch_size=10,noise_level=0,
                   blind_epochs=500)
    test_folds = 5
    valid_folds = 3
    dataopt = DataOpts(test_folds=test_folds,
                       valid_folds=valid_folds)
    res_all = []
    for i in range(test_folds):
        res = []    
        dataopt.test_fold_id=i
        testae = makeTest(dataopt=dataopt,traineropt=TrainerOpts(name='AE',
                      aeopt=aeopt))
        for k in range(valid_folds):    
            errs = testae.validate_fold(k)
            res.append(errs)
        errs = testae.test()
        res.append(errs)
        res_all.append(res)
    return res_all

def testall(t0):
    res = []
    for i in range(10):
        print 'Fold',i
        aere,aepe,aepen = t0.validate_fold(i)
        res.append([aere,aepe,aepen])
    aere,aepe,aepen = t0.test()
    res.append([aere,aepe,aepen])
    return res

def sample_run():
    dataopt = DataOpts(test_folds=5,test_fold_id=0)
    aeopt=GAEOpts(w=9,d=2,lam=1,l2=1e-1,top=None,noise_level=0,wr=1,
              decoder_l2r=0,robust_lambda=0,
              verbose=1,learning_rate_ratio=1,check_frequency=100)
    t = makeTest(traineropt=TrainerOpts(name='AE',aeopt=aeopt),dataopt=dataopt)
    res_ae = t.test()
    return res_ae
    
if __name__=='__main__': 
    res = setTest(5)    
#    main()
