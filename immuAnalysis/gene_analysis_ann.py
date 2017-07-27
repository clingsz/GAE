# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 20:06:28 2017

@author: cling
"""

# gene_analysis_for_annSinSam
import numpy
import immuAnalysis.clustering as cc
import immuAnalysis.gene_tools as gt
import misc.data_gen as dg
import misc.utils as utils
import matplotlib.pyplot as plt

def get_data_ann():
    gene_file = 'data/data.annSinSam.txt'
    x = numpy.loadtxt(gene_file,delimiter='\t',skiprows=1,usecols=range(1,9))
    x,_,_ = dg.make_standardize(x.transpose())
    x = x.transpose()
    Ds = numpy.loadtxt(gene_file,delimiter='\t',dtype='string',skiprows=0)    
    patientNames = Ds[0,1:] # first column is data source
    geneNames = Ds[1:,0]
    return x,geneNames,patientNames

def test():
    x,geneNames,patientNames = get_data_ann()
    cids,counts,mses = cc.gap_cluster(x)
    utils.saveobj('temp/ann_cluster.pkl',(cids,counts,mses))
    msigdb,msiglsts,msigpvs = gt.annotate_modules_with_msig(geneNames,cids)
    utils.saveobj('temp/ann_msig.pkl',(msigdb,msiglsts,msigpvs))

def summarize():
    x,geneNames,patientNames = get_data_ann()    
    cids,counts,mses = utils.loadobj('temp/ann_cluster.pkl')
    msigdb,msiglsts,msigpvs = utils.loadobj('temp/ann_msig.pkl')
    # 1. overall plot    
    cc.show_cluster(x,cids,mses,patientNames)
    plt.savefig('fig/ann_cluster/all_clusters.pdf',bbox_inches='tight')    
    # 2. clusterInfo
    write_cluster_info('fig/ann_cluster/',geneNames,cids,counts,mses,msigdb,msiglsts,msigpvs)

def write_cluster_info(filepath,geneNames,cids,counts,mses,msigdb,msiglsts,msigpvs):
    # write genes
    K = len(cids)
    TOP = len(msiglsts[0])
    for i in range(K):
        fileName = filepath + 'ann_cluster_' + str(i) + '.csv'
        print 'writing info to ' + fileName
        with open(fileName,'w') as f:
            testGenes = geneNames[cids[i]]
            f.write('ClusterID,NumOfGenes,MSE\n')
            f.write('%d,%d,%.2f\n' % (i,counts[i],mses[i]))
            f.write('Genes in the cluster:\n')
            utils.write_a_list(f,testGenes)
            f.write('Enriched MsigSet,URL,pvalue,sharedGene#,msigGene#,sharedGeneSet\n')
            for t in range(TOP):
                j = msiglsts[i][t]
                refGenes = msigdb[j]['genes'] 
                common = numpy.intersect1d(testGenes,refGenes)
                f.write('%s,%s,%.2e,%d,%d,' % (msigdb[j]['name'],msigdb[j]['url'],msigpvs[i][t],len(common),len(refGenes)))
                utils.write_a_list(f,common)
                