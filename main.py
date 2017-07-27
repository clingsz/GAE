# -*- coding: utf-8 -*-
"""
Created on Wed May 10 12:38:40 2017

@author: cling
"""

def summarizing_cross_validation():
    import misc.cv.collect_ND5_3 as cv
    cv.fig_boxplot_cverr()

def test_trainer():    
    import misc.data_gen as dg
    import gae.model.trainer as tr    
    data = dg.get_training_data()
    model = tr.build_gae()
    model.train(data['X'],data['Y'])

def plot_immune_age():
    import test.visualizations as vis
    vis.plot_immune_age()

def plot_MIGS():
    import test.visualizations as vis
    vis.plot_cyto_age(cyto_name = 'IL6')
#    vis.plot_cyto_age(cyto_name = 'IL1B')
#    vis.Jacobian_analysis()

def distribution_test():
    import immuAnalysis.distribution_test as dt
    dt.show_hist(range(3,53),'dist_cyto.pdf')
    dt.show_hist(range(53,78),'dist_cell.pdf')

def cell_cluster():
    import immuAnalysis.cytokine_clustering as cc
##    cc.main()
##    cc.pvclust_main()
    cc.agclust_main()
#    B = [10,20,50,100,1000]
##    cc.gap_stats(B)
#    import gfmatplotlib.pyplot as plt
#    plt.figure(figsize=[5*4,5])
#    for i in range(5):
#        plt.subplot(1,5,i+1)
#        cc.show_gapstats(B[i])
#    plt.tight_layout()
#    plt.show()
    
#    cc.generate_data_for_pvclust()
#    cc.choose_cluster()

#import immuAnalysis.module_genes as mg
if __name__ == '__main__':
	test_trainer()
#    summarizing_cross_validation()
#    import immuAnalysis.clustering as c
##
#    c.test()
#    import immuAnalysis.gene_analysis_ann as g
#    g.summarize()
    
