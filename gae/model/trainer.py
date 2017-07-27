# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 21:32:09 2017

@author: Tianxiang Gao

"""
#import GAE.model.age
from gae.model.guidedae import GAEOpts,fitGAE
from sklearn.linear_model import ElasticNetCV,Ridge
import timeit
import numpy
from sklearn.decomposition import PCA

def build_gae(w=5,d=2,wr=1,lam=0.2,l2=1e-3,
                    batch_size=10,noise_level=0,
                    verbose=1,blind_epochs=100,top=None):
    aeopt = GAEOpts(w=w,d=d,wr=wr,decoder_l2r=0,
                   l2=l2,noise_level=noise_level,
                   verbose=verbose,batch_size=batch_size,
                   blind_epochs=blind_epochs,
                   lam=lam)
    traineropt = TrainerOpts(name='GAE',aeopt=aeopt)
    return make_trainer(traineropt)

def vcopy(x):
    return x
    
class TrainerOpts(object):
    def __init__(self,name='None',n_components=5,aeopt=None):
        self.name = name
        self.n_components = n_components
        if aeopt is None:
            aeopt = GAEOpts()
        self.aeopt = aeopt

def mse(x,px):
    return numpy.mean((x-px)*(x-px))

def mae(x,px):
    return numpy.mean(numpy.abs(x-px))
    

class Trainer():
    def __init__(self,traineropt=None):
        if traineropt is None:
            traineropt = TrainerOpts()
        self.has_trained = False
        self.traineropt = traineropt
        self.name = traineropt.name

    def train(self,x,y):
        self.enet = ElasticNetCV(random_state=0,cv=5)   
        self.enet.fit(x,y.ravel())
        
    def test(self,vx,vy,vnx=None):
        if vnx is None:
            vnx = vx
        rce,rcae = self.get_dne(vx,vx)
        dne,dnae = self.get_dne(vnx,vx)
        pre,prae = self.get_pre(vx,vy)
        pren,praen = self.get_pre(vnx,vy)
        err = [rce,dne,pre,pren,rcae,dnae,prae,praen]
        print 'RCE: %.4f, APE: %.4f' % (rce,pre)
        return err
        
    def encode(self,x):
        return vcopy(x)
    def denoise(self,x):
        return vcopy(x)
    def predict(self,x):
        py = self.enet.predict(x)        
        py = py.reshape([py.shape[0],1])
        return py 
    def get_dne(self,x,y):
        py = self.denoise(x)
        return mse(y,py),mae(y,py)
    def get_pre(self,x,y):
        py = self.predict(x)
        return mse(py,y),mae(py,y)    

class Ridge_trainer(Trainer):
    def __init__(self,traineropt):
        self.l2 = 1e-3
        traineropt.name = 'Ridge'
        Trainer.__init__(self,traineropt)
    def train(self,x,y):
        self.enet = Ridge(alpha=self.l2)
        self.enet.fit(x,y.ravel())

def get_Jacobian(trainer,x,eps=1e-3):
    n,p = x.shape
    J = numpy.zeros([n,p])
    for i in range(p):
        x0 = x.copy()
        x1 = x.copy()
        x0[:,i] = x0[:,i] - eps
        x1[:,i] = x1[:,i] + eps
        J[:,i:i+1] = (trainer.predict(x1) - trainer.predict(x0))/(2.0*eps)
    return J

def get_RCE_residual(trainer,x):
    xr = trainer.denoise(x)
    E = numpy.square(xr-x)
    return E

def binarize(ty):
    ny = ty - numpy.min(ty)
    ny = ny/numpy.max(ny)
    return ny
    
def isBinary(ty):
    if len(numpy.unique(ty))==2:
        return True
    return False

class PCA_trainer(Trainer):
    def __init__(self,traineropt):
        n_components = traineropt.n_components
        self.n_components = n_components
        traineropt.name = 'PCA'+str(n_components)
        Trainer.__init__(self,traineropt)
    def encode(self,x):
        return self.pca.transform(x)
    def denoise(self,x):
        c = self.encode(x)
        y = self.pca.inverse_transform(c)
        return y
    def predict(self,x):
        c = self.encode(x)
        py = self.enet.predict(c)
        py = py.reshape([py.shape[0],1])
        return py 
    def train(self,x,y):
        n_components = self.n_components
        if n_components is None:
            n_components = x.shape[1]
        else:
            n_components = min(n_components,x.shape[1])
        self.pca = PCA(n_components=n_components)
        self.pca.fit(x)
        self.enet = ElasticNetCV(random_state=0,cv=10)   
        c = self.encode(x)
        self.enet.fit(c,y.ravel())
        self.has_trained = True

class GAE_trainer(Trainer):
    def __init__(self,traineropt):
        self.aeopt = traineropt.aeopt
        traineropt.name = 'GAE'+str(self.aeopt.lam)        
        if self.aeopt.lam==0:
            self.aeopt.train_on_enet = True
        Trainer.__init__(self,traineropt)
    def train(self,x,y):
        z = y
        time_start = timeit.default_timer()                    
        self.denoise,self.encode,epochs,self.predz = fitGAE(x,z,x,z,self.aeopt)
        if self.aeopt.train_on_enet:
            self.enet = ElasticNetCV(random_state=0,cv=5)
            self.enet.fit(self.encode(x),y.ravel())
        used_time = timeit.default_timer() - time_start
        print 'Training used %ds, %d epochs' % (used_time,epochs)                                         
        self.has_trained = True
        
    def predict(self,x):
        if self.aeopt.train_on_enet:
            y = self.enet.predict(self.encode(x))
        else:
            y = self.predz(x)
        return y.reshape([y.shape[0],1])

def make_trainer(traineropt=None):
    if traineropt is None:
        traineropt = TrainerOpts()
    train_model = traineropt.name
    if train_model=='GAE':
        model = GAE_trainer(traineropt)
    elif train_model=='None':
        model = Trainer(traineropt)
    elif train_model=='PCA':
        model = PCA_trainer(traineropt)
    elif train_model=='Ridge':
        model = Ridge_trainer(traineropt)
    return model

def build_linear():
    return make_trainer()

def build_pca(n_components=2):
    to = TrainerOpts(name='PCA',n_components=n_components)
    return make_trainer(to)

def build_ridge(l2=1e-3):
    ridge = make_trainer(TrainerOpts(name='Ridge'))
    ridge.l2 = l2
    return ridge
    