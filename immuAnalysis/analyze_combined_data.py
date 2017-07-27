# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 23:53:19 2017

@author: cling
"""

# analyze combined data
import gae.misc.data_gen as dg
import matplotlib.pyplot as plt
import numpy
import gae.misc.utils
from scipy.stats.mstats import normaltest
import gae.model.trainer
from sklearn.linear_model import LassoLars,Lars
import fitter

def normalityTest(D):
    xs,ms,vs = data_gen.make_standardize(D)
    if numpy.sum(numpy.isnan(xs))>0:
        print 'NaN in the data'
    k2,pv = normaltest(xs)
    return k2,pv

def get_data(chex_correction=2):
    return dg.Combine_dataset_2(dg.DataOpts(test_folds=5,chex_correction=chex_correction))

def check_no_nan(x):
    return numpy.sum(~numpy.isnan(x),axis=0)

def get_stats(x):
    # min,1stqt,median,3rdqt,max, mean, standard-deviation
    lst = numpy.where(~numpy.isnan(x[:,0]))[0]
    sx = x[lst,:]
    
    stats_info = ['min','1stqrt','median','3rdqrt','max','mean','std','valid']
    lls,dn = distribution_fit([1,2])
    stats_info = stats_info + dn
    stats = numpy.zeros([len(stats_info),sx.shape[1]])
    stats[0,:] = numpy.min(sx,axis=0)
    stats[1,:] = numpy.percentile(sx,0.25,axis=0)
    stats[2,:] = numpy.median(sx,axis=0)
    stats[3,:] = numpy.percentile(sx,0.75,axis=0)
    stats[4,:] = numpy.max(sx,axis=0)
    stats[5,:] = numpy.mean(sx,axis=0)
    stats[6,:] = numpy.std(sx,axis=0)
    stats[7,:] = sx.shape[0]
#    k2,pv = normalityTest(sx)
    for i in range(sx.shape[1]):
        print i
        xx = sx[:,i]
        lls,dn = distribution_fit(xx)
        lls = numpy.asarray(lls)
        stats[8:,i] = lls
#    stats_info = stats_info + dn
#    stats[8,:] = k2
#    stats[9,:] = pv
    return stats,stats_info
    
def table0_standard_feature_statistics():
    c = get_data()
    stats_cyto,si = get_stats(c.D_cyto)
    stats_flow,si = get_stats(c.D_flow)
    stats_age,si = get_stats(c.age)
    stats_bmi,si = get_stats(c.bmi)
    stats_mfs,si = get_stats(c.mfs)
    X = numpy.concatenate([stats_age,stats_bmi,stats_mfs,stats_cyto,stats_flow],axis=1)
    cn = ['Age','BMI','MFS'] + c.cyto_names.tolist() + c.flow_names.tolist()
    rn = si
    fname = 'tables/Table0_stats.csv'
    utils.save_csv_table(fname,X,cn,rn)
#    numpy.savetxt('tables/Table0_stats.csv',X,fmt='%.2f',header=header)
def fig0_nan_values():
    c = get_data()
    print 'Valid MFS', check_no_nan(c.mfs[:,0])
    print 'Valid BMI', check_no_nan(c.bmi[:,0])
    print 'Valid CMV', check_no_nan(c.cmv[:,0])
    print 'Valid EBV', check_no_nan(c.ebv[:,0])
    print 'Valid GENDER', check_no_nan(c.gender[:,0])    
    print 'Valid AGE', check_no_nan(c.age[:,0])    
    print 'Valid Flow_cyto:', check_no_nan(c.lD_flow[:,1])
    plt.imshow(numpy.isnan(c.lD_flow),aspect='auto',interpolation='none')
    plt.show()

def fig1_Cytokine_overall_boxplot():
    c1 = dg.Combine_dataset_2(dg.DataOpts(test_folds=5,chex_correction=False))    
    c2 = get_data()
    cs = [c1,c2]
    tits = ['Before CHEX Correction','After CHEX Correction']
    plt.figure(figsize=[10,10])
    for i in range(2):
        plt.subplot(2,1,i+1)
        c = cs[i]
        plt.boxplot(cs[i].x[:,:50])
        plt.xticks(numpy.arange(c.x.shape[1])+1,c.x_labels,rotation='vertical')
        plt.title(tits[i])
        plt.grid()
    plt.tight_layout()
    plt.show()


def fig2_Cytokine_hist():
    c = get_data()
    col = 6
    row = 9
    plt.figure(figsize=[col*1.5,row*1.5])
    stats_cyto,si = get_stats(c.x)
    lp = numpy.log(stats_cyto[-1,:])    
    for i in range(50):
        plt.subplot(row,col,i+1)
        plt.hist(c.x[:,i])
        plt.title(c.x_labels[i] + ('(%.1f)' % (lp[i])))
    plt.tight_layout()
    plt.show()



def fig3_PCA_Cytokine():
    c = get_data(2)
    x = c.x
    pca = trainer.get_pca_trainer(3)
    ds = c.get_cv_task(0)
    pca.train(ds.get_all())
    code = pca.encode(x)
    plt.figure(figsize=[8,8])
    plt.scatter(code[:,0],code[:,1],s=20,c=c.source,cmap='hot')
    plt.show()


from scipy.cluster.vq import kmeans2

def fig4_show_cytokines():
    c = get_data(2)
    ds = c.get_cv_task(0).get_all()
    x = ds[0][0]
    numpy.random.seed(0)
    center,inds = kmeans2(x,5)
    lst = numpy.argsort(inds)
    c2,ind2 = kmeans2(x.transpose(),5)
    plst = numpy.argsort(ind2)
    sx = x[lst,:]    
    plt.figure(figsize=[12,8])
#    plt.subplot(1,2,1)
#    plt.imshow(c.mfs[lst],aspect='auto',interpolation='none',
#               cmap='PRGn')
#    plt.subplot(1,2,2)
    plt.imshow(sx[:,plst],aspect='auto',interpolation='none',
               cmap='PRGn',vmax=4,vmin=-4)
    p = sx.shape[1]
    plt.ylabel('Patients')
    plt.xlabel('Cytokines')
    plt.xticks(numpy.arange(p),c.x_labels[plst],rotation='vertical')
    plt.colorbar()
#    plt.tight_layout()    
    plt.show()
    
def make_test(tr,ds):
    x,y,nx = ds[1]
    tr.train(ds)
    return tr.test(ds)
#    plt.scatter(tr.predict(x),y)
#    plt.show()
def fit_bz(bz,ds):
    ff = trainer.get_gae_trainer(w=bz,d=2,wr=1,
                                 l2=0.1,batch_size=5,
                                 verbose=1,lam=10,
                                 blind_epochs=1000,
                                 noise_level=0.3,top=0)
    return make_test(ff,ds)
    
def fig5_fit_age():
    dopt = dg.DataOpts(test_folds=5,target='AGE')
    c = dg.Combine_dataset_2(dopt)
    ds = c.get_cv_task(0).get_test()
    raw = trainer.get_raw_trainer()
    raw_res = make_test(raw,ds)   
    nn_res = []
    bzlst = [5]
#    bzlst = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    for bz in bzlst:
        nn_res.append(fit_bz(bz,ds))
    utils.saveobj('temp/fitagefd.pkl',(raw_res,nn_res))

def fig5_show_res():
    raw_res,nn_res = utils.loadobj('temp/fitagefd.pkl')
    nn_res = numpy.asarray(nn_res)
    plt.plot(nn_res[:,2])
    r = raw_res[2]
    plt.plot([0,nn_res.shape[0]],[r,r])

def fig6_bmi_and_age():
    c = get_data()
    bmi = c.bmi
    lst = numpy.where(~numpy.isnan(bmi))[0]
    bmi = bmi[lst]
    age = c.age[lst]
    plt.scatter(age,bmi)
    plt.show()
    
def fig7_mfs_threhold():
#    dopt = dg.DataOpts(test_folds=5,target='MFS')
#    c = dg.Combine_dataset_2(dopt)
#    ds = c.get_cv_task(0)
#    x,mfs,nx = ds.get_all()[0]
#    plt.hist(mfs)
    c = get_data()
    mfs = c.bmi
    lst = numpy.where(~numpy.isnan(mfs))[0]
    mfs = mfs[lst,:]
    x = c.x[lst,:]
    plt.hist(mfs,16)
    plt.show()
    for T in [20,25,30]:
        y = (mfs>T).astype('int')
#        c = utils.corr(x,y)
#        i = numpy.argmax(abs(c))
        py = trainer.logistic_regression(x,y)
#        print x.shape,y.shape,py.shape
        plt.scatter(py[:,1],y)
        plt.show()
        
def fig8_lars():
    test_folds = 10
    do = data_gen.DataOpts(test_folds=test_folds)
    c = data_gen.Combine_dataset_2(do)
    raw = trainer.IS_trainer()
    ts = range(1,11) + [15,20,30,40,50]
    n_ts = len(ts)
    E = numpy.zeros([test_folds,n_ts,2])
    for fold in range(test_folds):
        for top_id in range(n_ts):
            ds = c.get_cv_task(fold).get_test()
            e = raw.train_and_test(ds,ts[top_id])
            E[fold,top_id,:] = e
    utils.saveobj('temp/raw_E.pkl',E)
#    x = numpy.mean(E[:,:,0],axis=0)
#    y = numpy.mean(E[:,:,1],axis=0)
#    xs = numpy.std(E[:,:,0],axis=0)
#    ys = numpy.std(E[:,:,1],axis=0)
##    plt.scatter(E[:,0],E[:,1])
#    plt.errorbar(x,y,xs,ys)

def fig8_plot():
    E = utils.loadobj('temp/raw_E.pkl')
    ym = numpy.min(E[:,:,1],axis=1)
    E[:,:,1] = E[:,:,1] - ym.reshape([ym.shape[0],1])       
#    x = numpy.mean(E[:,:,0],axis=0)
    y = numpy.mean(E[:,:,1],axis=0)
    
    xs = numpy.std(E[:,:,0],axis=0)
    ys = numpy.std(E[:,:,1],axis=0)
#    plt.errorbar(x,y,xs,ys)
#    E = E.transpose([1,0,2])
#    plt.plot(E[:,:,0],E[:,:,1])
    ts = range(1,11) + [15,20,30,40,50]
    n_ts = len(ts)
    xt = numpy.arange(n_ts)+1
    plt.boxplot(E[:,:,1])
    plt.plot(xt,y,'-o')    
    plt.xticks(xt,ts)

if __name__ == '__main__':
#    s,si = get_stats(numpy.random.randn(50,3))
#    print s
#    distribution_fit_trial()
#    ae_train()
#    fig8_lars()
#    fig8_plot()
#    fig7_mfs_threhold()
    table0_standard_feature_statistics()
#    fig3_PCA_Cytokine()
#    fig4_show_cytokines()
#    fig5_fit_age()
#    fig5_show_res()
#    x = numpy.random.randn(3,2)
#    rn = ['1','2','3']
#    cn = ['A','B']
#    utils.save_csv_table('test.csv',x,cn,rn)
#    fig0_nan_values()
#    fig1_Cytokine_overall_boxplot()
#    fig2_Cytokine_hist()
#    fig6_bmi_and_age()