# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 19:26:32 2017

@author: cling
"""
import numpy
from scipy.stats import hypergeom
# Gene analysis tools

def get_enrichment_info(geneNames,cids,msigdb,msiglsts,msigpvs,i,k):
    testgenes = geneNames[cids[i]]
    j = msiglsts[i][k]
    refgenes = msigdb[j]['genes'] 
    common = numpy.intersect1d(testgenes,refgenes)
    msigName = msigdb[j]['name']
    msigurl = msigdb[j]['url']
    msigpv = msigpvs[i][k]
    commonLens = len(common)
    msigLens = len(refgenes)
    return (msigName,msigurl,msigpv,commonLens,msigLens)
    
def annotate_modules_with_msig(geneNames,cids,TOP=5):
    print 'loading msig dbs...'
    msigdb = loadMsig(geneNames)
    M = len(geneNames)
    msiglsts = []
    msigpvs =[]
    print 'identifying significant msigdbs...'
    for i in range(len(cids)):
        print i,len(cids)
        testgenes = geneNames[cids[i]]
        msiglst,msigpv = annotate(testgenes,msigdb,M)
        msiglsts.append(msiglst[:TOP])
        msigpvs.append(msigpv[:TOP])
        print msiglst[:TOP],msigpv[:TOP]
    return msigdb,msiglsts,msigpvs

def hygeotest(testSet,refSet,M):
    N = len(testSet)
    n = len(refSet)
    x = len(numpy.intersect1d(testSet,refSet))
    prb = 1-hypergeom.cdf(x-1, M, n, N)
    return prb

def annotate(testgenes,msigdb,M):
    msigpv = []
    for gs in msigdb:
        refgenes = gs['genes']        
        pv = hygeotest(testgenes,refgenes,M)
        msigpv.append(pv)
    msiglst = numpy.argsort(msigpv)
    return msiglst,sorted(msigpv)
    
def loadMsig(geneNames):
    filePath = 'data/msigdb.v6.0.symbols.gmt'
    msig = []
#    G,A,ylbs = get_gene_age_mappings()    
    gs = []
    with open(filePath,'r') as f:
        for line in f:
            l = line[:-1]
            temp = l.split('\t')
            geneset = {}
            geneset['name'] = temp[0]
            geneset['url'] = temp[1]        
            geneset['genes'] = temp[2:]
            gint = numpy.intersect1d(geneset['genes'],geneNames)
            geneset['genes'] = gint
            if len(gint)>0:
                msig.append(geneset)                
                gs.extend(geneset['genes'])
    intg = numpy.intersect1d(gs,geneNames)
    print 'testGenes: %d, msigCommonGenes: %d.' % (len(geneNames),len(intg))
    print len(msig), ' msig groups will be used (have intersection with test genes)'
    return msig