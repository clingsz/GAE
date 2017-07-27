# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 14:24:58 2017

@author: cling
"""

# clustering genes and modules

import immuAnalysis.cytokine_clustering as cc
import numpy
import matplotlib.pyplot as plt
import misc.utils as utils
from sigclust import recclust
from scipy.stats import ttest_ind

def get_genes():
#    gene_file = 'data/montoya_genes_centered.csv'
    gene_file = 'data/montoya_genes_raw.csv'
    D = numpy.loadtxt(gene_file,delimiter=',',skiprows=1)
    Ds = numpy.loadtxt(gene_file,delimiter=',',dtype='string',skiprows=0)    
    feature_names = Ds[0,:] # first column is data source
    patient_names = D[:,0:1]
    yids = range(1,5997)
    xids = range(5997,len(feature_names))
    Y = D[:,yids]
    X = D[:,xids]
    Y_names = feature_names[yids]
    X_names = feature_names[xids]
    data = {}
    data['patient_id'] = map(lambda x: str(int(x)), patient_names[:,0].tolist())
    data['X'] = X
    data['Y'] = Y
    data['X_labels'] = X_names
    data['Y_labels'] = Y_names
    return data

def draw_bound(bounds,x):
    nb = 0
    for b in bounds:
        nb += b
        plt.plot([nb,nb],[-1,x.shape[0]],'k--',markerSize=10)
    plt.xlim([0-0.5,x.shape[1]-0.5])
    plt.ylim([0-0.5,x.shape[0]-0.5])

def show_clust_lst(Y,rlst,clst,bounds):
    rY = Y[rlst,:]
    cY = rY[:,clst]
    plt.figure(figsize=[10,10])
    plt.imshow(cY,aspect='auto',interpolation='none',vmin=-3,vmax=3,cmap='PRGn')
    draw_bound(bounds,cY)    
    plt.colorbar()
    

def simple_clust():
    data = get_genes()
    Y = data['Y']    
    rlst = cc.clust(Y,10)
    clst = cc.clust(Y.transpose(),10)
    show_clust_lst(Y,rlst,clst)
    
def run_recclust():
    data = get_genes()
    Y = data['Y']
    res = recclust.recclust(Y.transpose(),mc_iters=100,threshold=0.05)
    utils.saveobj('temp/recclust_genes.pkl',res)

#run_recclust()
def import_recclust():
    res = utils.loadobj('temp/recclust_genes.pkl')
    cids = []
    lst = [res]
    while len(lst)>0:
        now = lst[0]
        if type(now) is dict:
#            print now['pval']
            if now['ids'] is not None:
                cids.append(now['ids'].astype('int').tolist())
                print len(now['ids']), now['pval']
            else:
                lst.append(now['subclust0'])
                lst.append(now['subclust1'])
        lst.remove(lst[0])
    return cids
    
def see_recclust():
    data = get_genes()
    Y = data['Y']
    c_ids = import_recclust()
    bounds = []
    for c in c_ids:
        bounds.append(len(c))
    rlst = cc.clust(Y,10)
    lst = numpy.concatenate(c_ids)
    show_clust_lst(Y,rlst,lst,bounds)

from sklearn.linear_model import LassoLarsCV

def regress_regulator_on_modules():
    data = get_genes()
    Y = data['Y']
    X = data['X']
    c_ids = import_recclust()
#    M = []
    lasso = LassoLarsCV(cv=5)
    W = []
    A = []
    i = 0
    for c in c_ids:
        y = numpy.mean(Y[:,c],axis=1)
        lasso.fit(X,y)
        w = lasso.coef_
        d = (X - numpy.repeat(y.reshape(y.shape[0],1),X.shape[1],axis=1))
        a = numpy.mean(d ** 2,axis=0)
        i+=1
        print i,len(c_ids)
#        print a.shape,w.shape
        W.append(w)
        A.append(a)
    utils.saveobj('temp/regulator_modules.pkl',(W,A))
    

def get_networks():
    W,A = utils.loadobj('temp/regulator_modules.pkl')
#    plt.plot(numpy.sort(numpy.asarray(W)).transpose())
    threshold = 0.05
    W = numpy.asarray(W)
    data = get_genes()
    ylbs = data['Y_labels']
    xlbs = data['X_labels']
    c_ids = import_recclust()
    f = open('temp/regnet.txt','w')
    for i in range(len(c_ids)):
        f.write('ID: %d COUNT: %d\n' % (i,len(c_ids[i])))
        f.write('Genes:')
        for c in c_ids[i]:
            f.write(' %s' % (ylbs[c]))
        f.write('\n')
        f.write('Regulators:')
        w = W[i,:]
        lst = numpy.where(abs(w)>threshold)[0].tolist()
        for r in lst:
            f.write(' %s %.2f' % (xlbs[r],w[r]))
        f.write('\n')
        
def plot_modules():
    W,A = utils.loadobj('temp/regulator_modules.pkl')
    threshold = 0.05
    W = numpy.asarray(W)
    data = get_genes()
    ylbs = data['Y_labels']
    xlbs = data['X_labels']
    Y = data['Y']
    patient_lst = cc.clust(Y,10)
    X = data['X']
    Y = Y[patient_lst,:]
    X = X[patient_lst,:]
    c_ids = import_recclust()
#    for i in range(len(c_ids)):
    for i in range(10):
        print i
        c = c_ids[i]
        w = W[:,i]
        rlst = numpy.where(abs(W[:,i])>=threshold)[0].tolist()
        r = (len(c)+len(rlst)+2)/6+1        
        plt.figure(figsize=[r,5])        
        cols = len(c)+len(rlst)+4
        mY = numpy.mean(Y[:,c],axis=1)
        sX = numpy.concatenate([mY.reshape(mY.shape[0],1),X[:,rlst]],axis=1)
        plt.subplot2grid((1,cols),(0,0),colspan=len(rlst)+1)
        plt.imshow(sX,aspect='auto',interpolation='none',vmin=-3,vmax=3,cmap='PRGn')
        plt.title('Regulators '+str(len(rlst)))
        xbs = map(lambda i: '%s(%.2f)' % (xlbs[i],w[i]),rlst)
        xbs = ['Module Mean'] + xbs
        plt.xticks(range(len(rlst)+1),xbs,rotation='vertical')
        plt.ylabel('Cluster ' + str(i))

        plt.subplot2grid((1,cols),(0,len(rlst)+2),colspan=len(c))
        plt.imshow(Y[:,c],aspect='auto',interpolation='none',vmin=-3,vmax=3,cmap='PRGn')
        plt.title('Genes '+str(len(c)))
        plt.xticks(range(len(c)),ylbs[c],rotation='vertical')
        plt.yticks([],[])

#        plt.tight_layout()
#        plt.show()
        pic_name = 'gene_module_pics/module' + str(i) +'.png'
        plt.savefig(pic_name,
                    bbox_inches='tight')
        
def summarize():
    data = get_genes()
    print data['X'].shape,data['Y'].shape
    c_ids = import_recclust()
    print len(c_ids)
    
    
def image_heatmap(G,age,glbs,mse,tit):
    l = G.shape[1]
    H = (l/7)+1
#    print H
    plt.figure(figsize=[8,H])
    plt.imshow(G.transpose(),aspect='auto',interpolation='none',
               cmap='bwr',vmax=3,vmin=-3)
    agestr = map(lambda x:str(int(x)),age)
    agelst = range(0,len(age),10)
    ageticks = [agestr[i] for i in agelst]
    plt.xticks(agelst,ageticks,rotation='vertical')
    plt.yticks(range(0,len(glbs)),glbs)
    plt.title('MSE:%.2f' % (mse) + tit)
    plt.xlabel('Age')
    plt.ylabel('Metabolic Genes')    
    plt.colorbar()
    
    
def plot_boxagegroup(G,cid,l0,l1):
    Gc = G[:,cid]        
    G0 = Gc[l0,:].ravel()
    G1 = Gc[l1,:].ravel()
    plt.boxplot([G0,G1],labels=['<=40','>40'])
    plt.ylabel(str(len(cid))+ ' Genes')
    diff = numpy.mean(G0)-numpy.mean(G1)
    s,pval = ttest_ind(G0,G1)
    plt.title('D=%.2f,P:%.2e' % (diff,pval))    

import string
import misc.data_gen as dg

def get_gene_age_mappings():
    data = get_genes()
    meta_file = 'data/metabolicGenes.csv'
    meta_genes = numpy.loadtxt(meta_file, delimiter=',', skiprows=1, dtype='string')
    genes = numpy.concatenate((data['Y_labels'],data['X_labels']),axis=0)
    upper = lambda x: string.upper(x)
    genes_upper = map(upper,genes)
    meta_genes = map(upper,meta_genes)
    gene_list = numpy.nonzero(numpy.in1d(genes_upper,meta_genes))[0]
    gene_data = numpy.concatenate((data['Y'],data['X']),axis=1)
    pids = data['patient_id']
    ref_data = dg.get_processed_data()
    ref_pids = ref_data['demo']['patient_id']
    age = []
    p_list = []
    for i in range(len(pids)):
        lst = numpy.where(ref_pids==pids[i])[0]
        if len(lst)>0:
            age.append(ref_data['demo']['age'][lst[0],0])
            p_list.append(i)
    G = gene_data[p_list,:]
    age_srt_list = numpy.argsort(age)    
    age = sorted(age)
    G = G[age_srt_list,:]
    G = G[:,gene_list]
    ylbs = genes[gene_list]
    A = numpy.asarray(age).reshape(len(age),1)
    print '%d/%d/%d metabolic genes/all metabolic gene/all genes. ' % (len(ylbs),len(meta_genes),len(genes))
    print '%d patients were mapped to their age.' % (G.shape[0])
    return G,A,ylbs

def check_CD248(G,A,ylbs):
    gid = numpy.where(ylbs=='CD248')[0]
    plt.scatter(A[:,0],G[:,gid])    
    plt.xlabel('Age')
    plt.ylabel('CD248')
    c = utils.corr(A,G[:,gid])
    plt.title('Corr=%.2f' % (c[0,0]))
    plt.show()
    
def estimate_params(G,l0,l1,A,cids):
    cinfo = {}
    cinfo['group_diff'] = []
    cinfo['group_pv'] = []
    cinfo['cids'] = cids
    cinfo['mse'] = []
    cinfo['size'] = []
    cinfo['age_corr'] = []
    cinfo['age_pval'] = []
    for i in range(len(cids)):
        cid = cids[i] 
        cinfo['size'].append(len(cid))
        Gc = G[:,cid]        
        G0 = Gc[l0,:].ravel()
        G1 = Gc[l1,:].ravel()
        diff = numpy.mean(G0)-numpy.mean(G1)
        s,pval = ttest_ind(G0,G1)
        mse = numpy.mean((Gc - numpy.repeat(numpy.reshape(numpy.mean(Gc,axis=1),
                                                          [Gc.shape[0],1]),
                                                          Gc.shape[1],axis=1))**2)        
        cinfo['group_pv'].append(pval)
        cinfo['group_diff'].append(diff)
        cinfo['mse'].append(mse)
        As = numpy.repeat(A,len(cid),axis=1)
        x = numpy.reshape(As,[As.shape[0]*As.shape[1],1])
        y = numpy.reshape(Gc,[Gc.shape[0]*Gc.shape[1],1])
        c,p = utils.spearmancorr(x,y)
        cinfo['age_corr'].append(c)
        cinfo['age_pval'].append(p)
    return cinfo

def plot_cluster_info(cinfo,K):
    plt.figure(figsize=[8,12])
    keys = ['mse','size','group_pv','group_diff','age_corr','age_pval']
    for i in range(6):
        plt.subplot(3,2,i+1)
        plt.plot(cinfo[keys[i]][:K],'-o')
        plt.xlabel('clusters')
        plt.ylabel(keys[i])
    plt.tight_layout()

        
def plot_corr_vs_age(G,cid,A):
    As = numpy.repeat(A,len(cid),axis=1)
    G = G[:,cid]
    x = numpy.reshape(As,[As.shape[0]*As.shape[1],1])
    y = numpy.reshape(G,[G.shape[0]*G.shape[1],1])
    plt.scatter(x,y,s=5)
    c,p = utils.spearmancorr(x,y)
    plt.xlabel('Age')
    plt.ylabel('Gene Expression Level')
    plt.title('Corr=%.2f, Pval=%.2e' % (c,p))
    
def boxplot_group_ge(G,l0,l1,cinfo,i):
    cid = cinfo['cids'][i]
    diff = cinfo['group_diff'][i]
    pval = cinfo['group_pv'][i]
    Gc = G[:,cinfo['cids'][i]]        
    G0 = Gc[l0,:].ravel()
    G1 = Gc[l1,:].ravel()
    plt.boxplot([G0,G1],labels=['<=40','>40'])
    plt.ylabel(str(len(cid)) + ' Genes')
    diff = numpy.mean(G0)-numpy.mean(G1)
    plt.title('Diff=%.2f, Pval:%.2e' % (diff,pval))

def reorder(A,lst):
    B = []
    for i in range(len(A)):
        B.append(A[lst[i]])
    return B

def sort_cinfo_on(cinfo,k):
    ke = cinfo[k]
    lst = sorted(range(len(ke)),key=lambda x:ke[x])
    for key in cinfo.keys():    
        cinfo[key] = reorder(cinfo[key],lst)
    return cinfo

def write_a_list(f,testgenes):
        f.write(testgenes[0])
        for g in testgenes[1:]:
            f.write(',' + g)
        f.write('\n')
    

def write_cluster_info(cinfo,i,ylbs,msigdb,filepath,TOP=5):

    msigpv = []
    testgenes = ylbs[cinfo['cids'][i]]
    for gs in msigdb:
        refgenes = gs['genes']        
        pv = hygeotest(testgenes,refgenes,850)
        msigpv.append(pv)
    lst = numpy.argsort(msigpv)

    # write genes
    with open(filepath,'w') as f:
        f.write('ClusterID,NumOfGenes,MSE,GroupDiff,GroupPval,AgeCorr,AgePval\n')
        f.write('%d,%d,%.2f,%.2f,%.2e,%.2f,%.2e\n' % (i,cinfo['size'][i],cinfo['mse'][i],
                                                      cinfo['group_diff'][i],cinfo['group_pv'][i],
                                                    cinfo['age_corr'][i],cinfo['age_pval'][i]))
        f.write('Genes in the cluster:\n')
        write_a_list(f,testgenes)
        f.write('Enriched MsigSet,URL,pvalue,sharedGene#,msigGene#,sharedGeneSet\n')
        for i in range(TOP):
            j = lst[i]
            refgenes = msigdb[j]['genes'] 
            common = numpy.intersect1d(testgenes,refgenes)
            f.write('%s,%s,%.2e,%d,%d,' % (msigdb[j]['name'],msigdb[j]['url'],msigpv[j],len(common),len(refgenes)))
            write_a_list(f,common)
            
    
from scipy.stats import hypergeom

def hygeotest(testSet,refSet,M):
    N = len(testSet)
    n = len(refSet)
    x = len(numpy.intersect1d(testSet,refSet))
    prb = 1-hypergeom.cdf(x-1, M, n, N)
    return prb

def loadMsig():
    filePath = 'data/msigdb.v6.0.symbols.gmt'
    msig = []
    G,A,ylbs = get_gene_age_mappings()    
    gs = []
    with open(filePath,'r') as f:
        for line in f:
            l = line[:-1]
            temp = l.split('\t')
            geneset = {}
            geneset['name'] = temp[0]
            geneset['url'] = temp[1]        
            geneset['genes'] = temp[2:]
            gint = numpy.intersect1d(geneset['genes'],ylbs)
            geneset['genes'] = gint
            if len(gint)>0:
                msig.append(geneset)                
                gs.extend(geneset['genes'])
#                print len(gint),len(geneset['genes']),len(ylbs)
#    print len(numpy.intersect1d(gs,ylbs))
    return msig

def plot_gene_scatter(G,A,ylbs,l0,l1):
    plt.figure(figsize=[8,8])
    for i in range(len(ylbs)):
        cid = [i]
        Gc = G[:,cid]        
        G0 = Gc[l0,:].ravel()
        G1 = Gc[l1,:].ravel()
        diff = numpy.mean(G0)-numpy.mean(G1)
        s,pval = ttest_ind(G0,G1)
        As = numpy.repeat(A,len(cid),axis=1)
        x = numpy.reshape(As,[As.shape[0]*As.shape[1],1])
        y = numpy.reshape(Gc,[Gc.shape[0]*Gc.shape[1],1])
        c,p = utils.spearmancorr(x,y)
        px = diff
        py = c
        plt.scatter(px,py)
        if c<-0.2 or c>0.1:
            plt.text(px,py,ylbs[i])
    plt.plot([-0.4,0.8],[0,0],'k--')
    plt.plot([0,0],[-0.4,0.3],'k--')
    plt.xlim([-0.4,0.8])
    plt.ylim([-0.4,0.3])
    plt.title('Metabolic genes')
    plt.xlabel('Young vs Old Group Mean Difference')
    plt.ylabel('Correlation vs Age')
    

def plot_cluster_scatter(cinfo,K):
    plt.figure(figsize=[15,8])
    for i in range(K):
        x = cinfo['age_corr'][i]
        y = -numpy.log(cinfo['age_pval'][i])
        sz = cinfo['size'][i]*2
        mse = cinfo['mse'][i]
#        if mse<0.3:
        plt.scatter(x,y,s=sz,c=mse,vmax=0.5,vmin=0,cmap='gist_rainbow')
        plt.text(x,y,str(i))
    plt.colorbar()
#    plt.plot([-0.2,0.3],[0,0],'k--')
#    plt.plot([0,0],[-0.15,0.15],'k--')
#    plt.xlim([-0.2,0.3])
#    plt.ylim([-0.15,0.15])
    plt.title('Clusters of metabolic genes')
#    plt.xlabel('Young vs Old Group Mean Difference')
    plt.xlabel('Correlation vs Age')
    plt.ylabel('-log p-value')

def analyze_metabolic_genes():
    # load genes
    G,A,ylbs = get_gene_age_mappings()
    l0 = numpy.where(A[:,0]<=40)
    l1 = numpy.setdiff1d(range(A.shape[0]),l0)
    dir_path = 'result/meta_modules/'
    
    # Plot summarization of all genes
    plot_gene_scatter(G,A,ylbs,l0,l1)
    plt.savefig(dir_path+'summary/gene_summary.png', bbox_inches='tight')
    plt.close()
    
    # cluster genes into 50 clusters
    K = 50    
    print 'cluster genes into %d clusters' % (K)    
    cids = cc.ag_clust(G.transpose(),K)
    cids.append(range(G.shape[1])) 
    cinfo = estimate_params(G,l0,l1,A,cids)

    # Plot the cluster information    
    plot_cluster_scatter(cinfo,K)
    plt.savefig(dir_path+'summary/cluster_summary.png', bbox_inches='tight')
    plt.close()
    plot_cluster_info(cinfo,K)
    plt.savefig(dir_path+'summary/cluster_summary_2.png', bbox_inches='tight')
    plt.close()
    
    # Do Msig db enrichment test for each cluster
    msigdb = loadMsig()
    for i in range(K+1):
        print i
        cid = cinfo['cids'][i]
        image_heatmap(G[:,cid],A,ylbs[cid],cinfo['mse'][i],'')
        plt.savefig(dir_path + 'c'+str(i) + '_heat.png', bbox_inches='tight')
        plt.close()
        plt.figure(figsize=[8,4])
        plt.subplot(1,2,1)
        plot_corr_vs_age(G,cid,A)
        plt.subplot(1,2,2)
        boxplot_group_ge(G,l0,l1,cinfo,i)        
        plt.tight_layout()
        plt.savefig(dir_path + 'c'+str(i) + '_age.png', bbox_inches='tight')
        plt.close()        
        write_cluster_info(cinfo,i,ylbs,msigdb,dir_path+'c'+str(i)+'_info.csv')
    
