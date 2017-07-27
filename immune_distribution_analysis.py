# -*- coding: utf-8 -*-
"""
Created on Wed May 10 12:38:40 2017

@author: Tianxiang Gao
"""

# Immune feature distribution analysis
import immuAnalysis.distribution_test as dt
import os

fileName = 'result/temp/dist_test_log.pkl'
# run distribution test
if not os.path.isfile(fileName):
    dt.distribution_test_run()
    
# summarize test and fetch the best distribution
if not os.path.isfile('result/temp/best_dist_ids.pkl'):
    dt.distribution_summarize()

# show the distribution result
dt.show_hist(range(10))