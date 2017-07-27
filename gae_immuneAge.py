# -*- coding: utf-8 -*-
"""
Created on Wed May 10 12:38:40 2017

@author: Tianxiang Gao
# identification of immune age using cross_validation
"""

import misc.cv.collect_ND5_3 as cv


#%% 1. Show cross validation setting:
cv.show_cv_setting()

#%% 2. Show general pre/rec error
cv.fig_boxplot_cverr()

#%% 3. Best code length:
print "GAE code score: rce + pre"
best_code_length = cv.get_best_code_length()
print "select the code that has the non-significant increase"

#%% 4. Best model setting:
setting = cv.get_best_model_setting(best_code_length)
print 'width:%d, depth:%d, alpha:%.1f, L2:%.3f' % (setting[0],setting[1],
                                                   setting[2],setting[3])
                                                   
#%% 5. Get Immune Age:
model = cv.get_best_model()       
immune_age = cv.get_immu_age(model)                                             