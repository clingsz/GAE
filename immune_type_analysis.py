# -*- coding: utf-8 -*-
"""
Created on Wed May 10 12:38:40 2017

@author: cling
"""

# Immune type analysis
# get the immune type

import immuAnalysis.cytokine_clustering as cc
import os

# identify the best cluster number using gap stats
B = 1000
fileName = 'result/temp/gapstats_B'+str(B)+'.pkl'
if not os.path.isfile(fileName):
    cc.gap_stats(B)
K = cc.show_gapstats(B)

# perform agclust and save the result
cc.agclust_main(K)
