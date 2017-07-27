# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 21:52:04 2017

@author: cling
"""

# cluster data

import misc.data_gen as dg
import numpy
import misc.utils as utils
#from scipy import stats
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering as ag
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os
from scipy.stats import spearmanr
from sklearn.metrics import silhouette_score,silhouette_samples
from scipy.spatial.distance import pdist,squareform
import matplotlib
import misc.gap as gap

def PCA_correction(x,n_components=1):
    x = x.transpose()
    pca = PCA(n_components=n_components)
    pca.fit(x)
    codes = pca.transform(x)
    xr = pca.inverse_transform(codes)
    y = x - xr
    return y.transpose()

def spearmancorr(x,y):
    C,b = spearmanr(numpy.concatenate([x,y],axis=1))
    p1 = x.shape[1]
    return C[:p1,p1:],b[:p1,p1:]

def get_data(correction=True):
    data = dg.get_processed_data(correction)
    data_raw = dg.get_raw_combined_dataset()
    vlst = numpy.where(~numpy.isnan(numpy.sum(data['flow'],axis=1)))[0]
    demo = data['demo']
    demo_nms = ['source','age','gender','bmi','mfs']
    D = {}
    for nm in demo_nms:
        D[nm] = demo[nm][vlst,:]        
    x = data['cyto'][vlst,:]
    z = data['flow'][vlst,:]    
    xz = numpy.concatenate([x,z],axis=1)
    x,_,_ = dg.make_standardize(x)    
    z,_,_ = dg.make_standardize(z)
    source = demo['source']
    D['x'] = x
    D['z'] = z
    D['xz'] = xz
    D['xlbs'] = data['cyto_names']
    D['zlbs'] = data['flow_names']
    D['demo'] = demo_nms
    D['pids'] = demo['patient_id']
    D['flow'] = data_raw['D_flow'][vlst,:]
    corr,pv = spearmancorr(source[vlst,:],xz)
    return D,numpy.mean(pv<0.05),numpy.mean(abs(corr))

def clust(x,n):
    a = ag(n_clusters=n)
    a.fit(x)
    lbs = a.labels_
    lst = numpy.argsort(lbs)
    return lst

def find_best_cluster_n(x):
    res = []
    xx = squareform(pdist(x))
    k_ranges = range(2,10)
    for k in k_ranges:
        lbs = ag(n_clusters=k).fit_predict(x)
        print len(lbs),x.shape,xx.shape
        res.append(silhouette_samples(xx,lbs))
    res = numpy.asarray(res).transpose()
    plt.boxplot(res)
    plt.show()
    plt.plot(numpy.mean(res,axis=0))

def ag_clust(x,k):
    a = ag(n_clusters=k)
    a.fit(x)
    lbs = a.labels_
    cids = []
    for i in range(k):
        lst = numpy.where(lbs==i)[0]
        cids.append(lst)
#    lst = numpy.argsort(lbs)
    return cids

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
    
def show_cluster(D,cids=None,tit=None,mses=None):
    font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 10}
    matplotlib.rc('font', **font)
    if cids is None:
        cids = [[i for i in range(D['x'].shape[0])]]
#    Ds,nms,x,xlbs,c,z,zlbs = get_data()
    # x cyto, z flow
    x,xlbs = D['x'],D['xlbs']
    z,zlbs = D['z'],D['zlbs']

    plst = clust(x.transpose(),6)
    xs = x[:,plst]
    zlst = clust(z.transpose(),6)
    zs = z[:,zlst]
    K = len(cids)
    p = len(plst)
    zp = len(zlst)
    Cyto = []
    Flow = []
    phenos = [D[demo] for demo in D['demo']]
    PS = {}
    C = []
    bounds = []
    for i in range(K):
        bounds.append(len(cids[i]))
        xi = xs[cids[i],:].transpose()
        zi = zs[cids[i],:].transpose()
        C = C + [i]*len(cids[i])
        if i==0:
            Cyto = xi
            Flow = zi
        else:
            Cyto = numpy.concatenate([Cyto,xi],axis=1)
            Flow = numpy.concatenate([Flow,zi],axis=1)
        for j in range(len(phenos)):
           pi = phenos[j][cids[i],:].transpose()
           if i==0:
               PS[j] = pi
           else:
               PS[j] = numpy.concatenate([PS[j],pi],axis=1)
    C = numpy.asarray(C).reshape(1,len(C))
    plt.figure(figsize=[25,15])
#    n = C.shape[1]
    cyto_row = 0
    flow_row = 10
    other_rows = len(phenos)+1    
    rows = cyto_row + flow_row + other_rows
    phstr = D['demo']
    vmax = [1,90,1,40,10]
    vmin = [0,10,0,15,3]
    
    row_now = 0
    plt.subplot2grid((rows,1),(0,0),rowspan=1)
    plt.imshow(C,aspect='auto',interpolation='none',vmax=K,vmin=0,cmap='Set1')
    plt.colorbar()
    plt.yticks([0],['ClusterID'])
    plt.xticks([],[])
    if tit is not None:
        plt.title(tit)
    row_now += 1

    for j in range(len(phenos)):
        plt.subplot2grid((rows,1),(row_now,0),rowspan=1)
        xnow = PS[j]
        plt.imshow(xnow,aspect='auto',interpolation='none',vmax=vmax[j],vmin=vmin[j],cmap='bwr')
        draw_bound(bounds,xnow)
        plt.colorbar()
        plt.yticks([0],[phstr[j]])
        plt.xticks([],[])
        row_now += 1
    
#    plt.subplot2grid((rows,1),(row_now,0),rowspan=cyto_row)
#    plt.imshow(Cyto,aspect='auto',interpolation='none',vmax=3,vmin=-3,cmap='PRGn')
#    plt.colorbar()
#    plt.yticks(numpy.arange(p),xlbs[plst])
#    plt.xticks([],[])
#    row_now += cyto_row
#    draw_bound(bounds,Cyto)
    
    
    plt.subplot2grid((rows,1),(row_now,0),rowspan=flow_row)
    plt.imshow(Flow,aspect='auto',interpolation='none',vmax=3,vmin=-3,cmap='PRGn')
    plt.colorbar()
    plt.yticks(numpy.arange(zp),zlbs[zlst])
#    plt.xticks([],[])
    row_now += flow_row
    pos = draw_bound(bounds,Flow)
    if mses is not None:
        mses = numpy.round(mses,decimals=2)
        plt.xticks(pos,mses,rotation='vertical')
#    plt.show()

def plot_batch_effect_removal():
    pvrs = []
    cors = []
    for pc in range(10):
        D,pvr0,cor0 = get_data(True,pc)  
        D,pvr1,cor1 = get_data(False,pc)
        pvrs.append([pvr0,pvr1])
        cors.append([cor0,cor1])
    plt.figure(figsize=[10,5])
    for i,ts,ylb in zip([1,2],[pvrs,cors],['Significant BE ratio','Mean absCorr']):
        plt.subplot(1,2,i)
        plt.plot(ts)
        plt.ylabel(ylb)
        plt.xlabel('Top PCA Components')
        plt.legend(['Batch_Corrected','PCA Only'])

def generate_data_for_pvclust():
    D,pvr0,cor0 = get_data(True)
#    xz = numpy.concatenate([D['x'],D['z']],axis=1)
    utils.save_csv_table('data/for_pvclust.csv',D['z'])
    show_cluster(D)

def comparing_batch_correction():
    D,mpv,mcr = get_data(False)
    print mpv,mcr
    st = 'Before Correction, sigRatio:%.2f, meanCorr:%.2f' % (mpv,mcr) 
    show_cluster(D,tit=st)
    D,mpv,mcr = get_data(True)
    print mpv,mcr    
    st = 'After Correction, sigRatio:%.2f, meanCorr:%.2f' % (mpv,mcr)
    show_cluster(D,tit=st)

def choose_cluster():
    D,_,_ = get_data()
    z = D['z']
#    clst = [10,20,30,40,50]
    clst = range(20,40,20)
#    colors = ['y','r','m','b','g']
    plt.figure(figsize=[10,10])
    cs = []
    ms = []
    clrs = []
    for i in range(len(clst)):            
        cids = ag_clust(z,clst[i])
        c,m = analyze_cluster(z,cids)
        cs.extend(c)
        ms.extend(m)
        clrs.extend([clst[i]]*len(c))
    plt.scatter(cs,ms,c=clrs,cmap='hot',vmax=max(clst),vmin=0,s=100)
    plt.colorbar()
    plt.show()
    
def agclust_main(K):    
    D,_,_ = get_data()
    cids = ag_clust(D['z'],K)
    c,m = analyze_cluster(D['z'],cids)
    lst = sorted(range(K),key=lambda i:m[i])
    cjds = utils.reorder(cids,lst)
    cj = utils.reorder(c,lst)
    mj = utils.reorder(m,lst)
    show_cluster(D,cjds,tit='agclust k='+str(K),mses=mj)
    plt.savefig('result/fig/clusters/agclust_cells_'+str(K)+'.pdf',bbox_inches='tight')
    write_cluster_info(D,cjds,cj,mj,'result/tables/agclust_cells_'+ str(K) + '.csv')
    write_cluster_mean(D,cjds,cj,mj,'result/tables/agclust_cells_mean_raw_'+ str(K) + '.csv')
    
def gap_stats(B):
    D,_,_ = get_data()
    z = D['z']
#    z = init_board_gauss(100,10)
    ks = range(1,30)
    print z.shape
#    B = 1000
#    gaps = gap.gap(z, refs=None, nrefs=100, ks=ks,AGclust = True)
    ks, logWks, logWkbs, sk = gap.gap_statistic(z,B=B,ks=ks)
    utils.saveobj('result/temp/gapstats_B'+str(B)+'.pkl',(ks,logWks,logWkbs,sk))

def show_gapstats(B):
    ks,logWks,logWkbs,sk = utils.loadobj('result/temp/gapstats_B'+str(B)+'.pkl')    
    gaps = logWkbs - logWks
#    plt.plot(ks,logWks,'r')
#    plt.plot(ks,logWkbs,'b')
#    plt.show()
    gapinc = gaps[:-1]-gaps[1:]+sk[1:]
    plt.plot(ks[:-1],gapinc,'-o')
    plt.plot([min(ks),max(ks)],[0,0],'k--')
    plt.xlabel('Cluster number')
    plt.ylabel('Decision bound (>=0)')
    lst = numpy.where(gapinc>=0)[0]
    print ks[lst[0]]
    best_K = ks[lst[0]]
    plt.title('BS='+str(B)+', Best Cluster Number = ' + str(best_K))
    plt.grid()
    return best_K

def write_a_list(f,testgenes):
        f.write(testgenes[0])
        for g in testgenes[1:]:
            f.write(',' + g)
        f.write('\n')

def write_cluster_mean(D,cids,c,m,filepath='tables/immunotypes.csv'):
#    filepath = 'tables/immunotypes.csv'
    flowstr = ','.join(['ClusterID']+D['zlbs'].tolist())    
    with open(filepath,'w') as f:
        f.write(flowstr+'\n')
        for i in range(len(cids)):
#            ml = numpy.mean(D['z'][cids[i],:],axis=0)
            ml = numpy.mean(D['flow'][cids[i],:],axis=0)
            f.write('%d,' % (i))
            write_a_list(f,map(str,ml))
      
def write_cluster_info(D,cids,c,m,filepath='tables/immunotypes.csv'):
#    filepath = 'tables/immunotypes.csv'
    with open(filepath,'w') as f:
        f.write('ClusterID,NumOfPatients,MSE,PatientIDs\n')   
        for i in range(len(cids)):
#            print i
            f.write('%d,%d,%.2f,' % (i,c[i],m[i]))
            pids = []
            for j in cids[i]:
                pids.append(D['pids'][j])
            write_a_list(f,pids)            

if __name__ == '__main__':
    main()
    
