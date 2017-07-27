# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 10:53:51 2017

@author: cling
"""

# collect ND5_3

import misc.cv.exp_test as exp_test
from misc.utils import saveobj,getJobOpts,loadobj,spearmancorr
import numpy
from misc.data_gen import DataOpts,load_data
import misc.data_gen as data_gen
from gae.model.trainer import TrainerOpts
import gae.model.trainer as trainer
import cPickle
from gae.model.guidedae import GAEOpts
import misc.cv.run_exps as run_exps
import matplotlib.pyplot as plt
from scipy import stats
import os

# comparing the best model
def show_cv_setting():
    w_lst,d_lst,lam_lst,l2_lst = get_cv_setting()
    print 'width list:',w_lst
    print 'depth list:',d_lst
    print 'alpha list:',lam_lst
    print 'L2 list:',l2_lst
    print 'test folds = 5'
    print 'validation folds = 3'
    
     
    
def comparing_best_model():
    best_code_length = get_best_code_length()
    gaes,pars = loadobj('result/temp/ND5-3-AE-LAMW.pkl')
#    lam_lst = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    lam_lst = [0,0.1,0.3,0.5,0.7,1.0]
    gaes = filter(lambda g: g.paras[0][0]==best_code_length and g.paras[0][2] in lam_lst,gaes)
    plt.figure(figsize=[10,10])
    for g in gaes:
#        print g.name
        l = g.paras[0,2]
        g = exp_test.calc_best_test_err(g,[1-l,l])
        RE = g.best_test_errs[:,0]
        PE = g.best_test_errs[:,2]
        NM = g.name
        x,y = numpy.mean(RE),numpy.mean(PE)
        sx,sy = numpy.std(RE)/numpy.sqrt(5),numpy.std(PE)/numpy.sqrt(5)
        plt.errorbar(x,y,yerr=sy,xerr=sx)
        plt.text(x,y,NM)
    raw,pca = loadobj('result/temp/ND5-3-rawpca.pkl')
#    for i in range(pca.test_errs.shape[0]):
    for i in range(3,6):
        RE = pca.test_errs[i,:,0]
        PE = pca.test_errs[i,:,2]
        NM = 'PCA'+str(i+1)
        x,y = numpy.mean(RE),numpy.mean(PE)
        sx,sy = numpy.std(RE)/numpy.sqrt(5),numpy.std(PE)/numpy.sqrt(5)
        plt.errorbar(x,y,yerr=sy,xerr=sx)
        plt.text(x,y,NM)
    rawpe = numpy.mean(raw.best_test_errs[:,2])
    plt.plot([0.20,0.50],[rawpe]*2,'k--')        
    plt.show()

def get_best_model_setting(best_code_length):
    print 'Getting best model setting...'
    gaes,pars = loadobj('result/temp/ND5-3-AE-W.pkl')
    bgae = gaes[best_code_length-1] 
    print bgae.name
    terr = bgae.test_errs
    min_err_id = numpy.argmin(numpy.mean(terr[:,:,0] + terr[:,:,2],axis=1))
    return bgae.paras[min_err_id,:]

def get_immu_age(model):
    print 'getting Imunne Age for all patients'
    data = data_gen.get_processed_data()
    x,_,_ = data_gen.make_standardize(data['cyto'])
    ag,mu,sig = data_gen.make_standardize(data['demo']['age'])    
    y = model.predict(x)
    y = y * sig + mu
    immu_age = y
    plt.scatter(data['demo']['age'],immu_age)
    plt.xlim([0,100])
    plt.ylim([0,100])
    plt.plot([0,100],[0,100],'k--')
    plt.ylabel('Immune Age')
    plt.xlabel('Age')
    plt.show()
#    X = numpy.concatenate([data['demo']['age'],immu_age],axis=1)
#    utils.save_csv_table('tables/ia_vs_age.csv',X,col_name=['Age','ImmuAge'])
    return immu_age

def show_best_gae(model):
    data = load_data(DataOpts())
    x = data.get_all()[0][0]
    xlb = data.x_labels
    J = trainer.get_Jacobian(model,x)
    lst = numpy.argsort(-numpy.sum(numpy.abs(J),axis=0))
    plt.figure(figsize=[10,5])
    plt.boxplot(J[:,lst])
    plt.grid()
    plt.xticks(numpy.arange(len(xlb))+1,xlb[lst],rotation='vertical')
    plt.ylabel('Jacobian')
    plt.show()

def fig_boxplot_cverr():
    aes,nms = get_results()
    E = numpy.asarray(aes)
    pre_E = E[:,:,2]
    rce_E = E[:,:,0]
    n = E.shape[0]

    plt.figure(figsize=[10,10])
    xm = numpy.mean(rce_E,axis=1)
    xs = numpy.std(rce_E,axis=1)/5.0
    ym = numpy.mean(pre_E,axis=1)
    ys = numpy.std(pre_E,axis=1)/5.0
    plt.errorbar(xm,ym,xerr=xs,yerr=ys,fmt='o')
    
    for i in range(n):
        plt.text(xm[i],ym[i],nms[i])
    plt.grid()
    plt.xlabel('RCE')
    plt.ylabel('Age prediction')    
    plt.show()

def get_results():
    cverrs = []
    nms = []
    rawpcafile = 'result/temp/ND5-3-rawpca.pkl'
    if not os.path.isfile(rawpcafile):
        print 'did not find '+rawpcafile +' Run...'
        get_RAW_PCA_result()
    raw,pca = loadobj(rawpcafile)
    cverrs.append(raw.best_test_errs)
    nms.append('RAW')
    E = pca.test_errs
    for i in range(E.shape[0]):
        cverrs.append(E[i,:,:])   
        nms.append('PCA'+str(i+1))
    aefile = 'result/temp/ND5-3-AE-W.pkl'
    if not os.path.isfile(aefile):
        print 'did not find '+ aefile +' Run...'
        collect_ND5_3()
    gaes,pars = loadobj(aefile)
    for er in gaes:
        er = exp_test.calc_best_test_err(er,[0.5,0.5])
        cverrs.append(er.best_test_errs)
        nms.append(er.name)
    return cverrs,nms

def get_best_code_length():
    aes,nms = get_results()
    aes = aes[11:]
    print 'Getting best code length...'
    w_lst = range(1,11)
    c = 0
    m = len(w_lst)
    PE = numpy.zeros([m,5])
    pvs = []
    best_code_length = 1
    for i in range(m):
             PE[i,:] = aes[c][:,0] + aes[c][:,2]
             if i>0:
                _,pvpe = stats.ttest_rel(PE[i,:],PE[i-1,:])
                pvs.append(pvpe)
                print i,i+1,pvpe
                if best_code_length==1 and pvpe>0.05:
                    best_code_length = i
             c = c + 1
    plt.boxplot(PE.transpose())
    plt.title('Best Code Length: ' + str(best_code_length))
    plt.xlabel('Code Length')
    plt.ylabel('Total Loss')
    plt.show()
    return best_code_length
            
def get_best_model(fileName='result/temp/best_gae_model.pkl'):
    if os.path.isfile(fileName):
        model = loadobj(fileName)
    else:
        print 'Train on the best model settings...'
        best_model_setting = get_best_model_setting()
        w,d,lam,l2 = best_model_setting
        model = trainer.build_gae(w=int(w),d=int(d),lam=lam,l2=l2,verbose=1,
                                            wr=1,batch_size=10,noise_level=0,blind_epochs=500)
        
        print w,d,lam,l2
        data = data_gen.get_training_data()
        model.train(data['X'],data['Y'])
        saveobj(fileName,model)
    return model

def compare_age_prediction_error_without_MIG(best_model_setting,compageFile):
    w,d,lam,l2 = best_model_setting
    print w,d,lam,l2
    aeopt = GAEOpts(w=int(w),wr=1,d=int(d),
                   lam=lam,verbose=1,
                   l2=l2,batch_size=10,noise_level=0,
                   blind_epochs=500)
    test_folds = 5
    dataopt = DataOpts(test_folds=test_folds)
    gae = trainer.make_trainer(traineropt=TrainerOpts(name='AE',
                      aeopt=aeopt))                        
    
    e = []
    for t in range(test_folds):
        dataopt.test_fold_id = t
        data = load_data(dataopt)
        MIGid = numpy.where(data.x_labels=='MIG')[0][0]
        
        ds = data.get_test()
        errs1 = gae.train_and_test(ds)
        ds[0][0][:,MIGid] = 0
        ds[1][0][:,MIGid] = 0
        errs2 = gae.train_and_test(ds)
        e.append([errs1[2],errs2[2]])
    saveobj(compageFile,e)


def plot_IMage_vs_other_phenotypes(immu_age):
    data = data_gen.get_processed_data()
    demo = data['demo']
    pheno = [demo['age'],demo['bmi'],demo['mfs']]
    phnms = ['age','bmi','mfs']
    phenobin = [demo['source'],demo['gender'],demo['cmv'],demo['ebv']]
    phbins = ['source','gender','cmv','ebv']
    plt.figure(figsize=[10,4])
    for i,ph,pn in zip(range(1,4),pheno,phnms):
        plt.subplot(1,3,i)
        lst = numpy.where(~numpy.isnan(ph))[0]
        plt.scatter(ph[lst,:],immu_age[lst,:])
        a,b = spearmancorr(ph[lst,:],immu_age[lst,:])
        plt.ylabel('ImmuAge')
        plt.xlabel(pn)
        plt.title('SpCorr=%.2f,PV=%.3f' % (a,b))
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=[10,10])
    for i,ph,pn in zip(range(1,5),phenobin,phbins):
        plt.subplot(2,2,i)
        lst = numpy.where(~numpy.isnan(ph))[0]
        ph = ph[lst,:]
        y = immu_age[lst,:]
        l0 = numpy.where(ph==0)[0]
        l1 = numpy.where(ph==1)[0]
        plt.boxplot([y[l0],y[l1]])
        a,b = spearmancorr(ph,y)
        plt.ylabel('ImmuAge')
        plt.xlabel(pn)
        plt.title('SpCorr=%.2f,PV=%.3f' % (a,b))
    plt.tight_layout()
    plt.show()
    
from sklearn.decomposition import PCA

def pcatrans(x):
    pca = PCA(n_components=2)
    pca.fit(x)
    z = pca.transform(x)
    return z    
    
def showpcavsgae():
    best_model_name = 'result/temp/best_gae_model.pkl'
    model = loadobj(best_model_name)
    data = data_gen.get_processed_data()
    x,_,_ = data_gen.make_standardize(data['cyto'])
    age = data['demo']['age']
    c = model.encode(x)    
    z1 = pcatrans(x)
    z2 = pcatrans(c)
    plt.figure(figsize=[10,5])
    tits = ['PCA on Cytokines','PCA on GAE codes']
    for i,z in zip(range(2),[z1,z2]):
        plt.subplot(1,2,i+1)
        plt.scatter(z[:,0],z[:,1],c=age,cmap='hot',s=15)
        plt.xlabel('PCA-1')        
        plt.ylabel('PCA-2')
        plt.title(tits[i])
        plt.colorbar()
    plt.show()
    
def get_lin_model():
    test_folds = 5
    valid_folds = 3
    dataopt = DataOpts(test_folds=test_folds,
                       valid_folds=valid_folds)
    test = exp_test.makeTest(dataopt=dataopt,
                             traineropt=TrainerOpts(name='None'))
    model = test.trainer
    data = test.dataset.get_all()
    model.train(data)
    plt.scatter(data[0][1],model.predict(data[0][0]))      
    plt.show()      
    return model


def get_cv_setting():
    w_lst = range(1,11)
    d_lst = [1,2,3]     
    lam_lst = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0] 
    l2_lst = [1e-3,1e-2,1e-1]
    return w_lst,d_lst,lam_lst,l2_lst    

def collect_ND5_3():    
    file_name = 'result/temp/logs_ND5_3.pkl'
    res = cPickle.load(open(file_name,'rb'))    
    w_lst,d_lst,lam_lst,l2_lst = get_cv_setting()
    joblsts = [w_lst,d_lst,lam_lst,l2_lst]
    
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
        for w in w_lst:
            er = exp_test.ExpResult('GAE'+str(lam)+'-'+str(w))
            lim = numpy.where(paras[:,0]==w)[0]
            lst = numpy.where(paras[:,2]==lam)[0]
            lst = numpy.intersect1d(lim,lst)
            er.load_exp_result(valid_errs[lst,:,:,:],test_errs[lst,:,:],paras[lst,:])
            aes.append(er)
    saveobj('result/temp/ND5-3-AE-LAMW.pkl',[aes,(test_errs,paras)])

    aes = []
    for w in w_lst:
            er = exp_test.ExpResult('GAE'+'-'+str(w))
            lst = numpy.where(paras[:,0]==w)[0]
            er.load_exp_result(valid_errs[lst,:,:,:],test_errs[lst,:,:],paras[lst,:])
            aes.append(er)
    saveobj('result/temp/ND5-3-AE-W.pkl',[aes,(test_errs,paras)])

def get_RAW_PCA_result():
    dataopt = DataOpts(test_folds=5)
    blER = run_exps.get_raw_result(dataopt)
    pcaER = run_exps.get_pca_result(dataopt,lst=range(1,11))
    saveobj('result/temp/ND5-3-rawpca.pkl',[blER,pcaER])

def main():
    best_model_name = 'temp/best_gae_model.pkl'
#    if not os.path.isfile(best_model_name):
#        get_best_model(best_model_name)
    model = loadobj(best_model_name)
#    linmodel = get_lin_model()
    immu_age = get_immu_age(model)
#    show_best_gae(model)
#    plot_IMage_vs_other_phenotypes(immu_age)
#    compare_age_prediction_error_without_MIG()
        
def run():
    best_model_name = 'temp/best_gae_model.pkl'
    collect_ND5_3()
    train_best_model(best_model_name)

def get_model():
    best_model_name = 'temp/best_gae_model.pkl'
    return loadobj(best_model_name)    

if __name__ == '__main__':
    main()
#    showpcavsgae()
#    main()
#    comparing_best_model()
#    get_best_code_length()
#    fig_boxplot_cverr()
#    aes = get_results()    
#    E = numpy.asarray(aes)
#    get_results()
#    collect_ND5_3()
#    get_RAW_PCA_result()
#    get_gae_path()