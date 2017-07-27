# -*- coding: utf-8 -*-
"""
Created on Wed May 10 12:38:40 2017

@author: Tianxiang Gao
"""

# analysis guided-auto-encoder

# 3 experiment
# 1) train cytokine using PCA and AE (cv error, visualization)
# 2) AE with different initialization
# 3) GAE the path, visualization

import misc.data_gen as dg
import misc.utils as utils
import gae.model.learner as learner
import matplotlib.pyplot as plt
import numpy
from sklearn.linear_model import ElasticNetCV
########################################################
#  0. Load the data
########################################################
immune_data = dg.load_immune(folds=5)
x,y = immune_data['x_train'],immune_data['y_train']
xv,yv = immune_data['x_test'],immune_data['y_test']

#%%
########################################################
#  1. Run the AE and PCA on the training dataset
########################################################
result_file_name = 'result/temp/gae_analysis_result.pkl'
LOADRESULT = True
if not LOADRESULT:
    reses = []
    for i in range(5):
        res = learner.AE_train(x,y,alpha=0,randseed=i)
        reses.append(res)
    utils.saveobj(result_file_name,reses)
else:
    reses = utils.loadobj(result_file_name)
pcares = learner.PCA_train(x,n_components=2)
reses = [pcares] + reses
#%%
########################################################
#  2. plot explained variance %
########################################################

rs = []
r_title = []
i = 0
for res in reses:
    rs.append(learner.var_exp(x,res[0],res[1]))
    r_title.append('AE' + str(i))
    i += 1
r_title = ['PCA'] + r_title

plt.figure(figsize=[8,5])
plt.bar(numpy.arange(i)-0.5,rs)
plt.xticks(range(i),r_title)
plt.ylabel('Explained Variance')
plt.savefig('result/fig/pca_vs_ae_1.pdf', bbox_inches='tight')

plt.figure(figsize=[12,6])
i = 0
for i in range(6):
    plt.subplot(2,3,i+1)
    res = reses[i]
    learner.visualizer(xv,res[0],res[1],r_title[i])
plt.tight_layout()
plt.savefig('result/fig/pca_vs_ae_2.pdf', bbox_inches='tight')

#%%
########################################################
#  3. The difference between codes
########################################################
cs = []
xx = numpy.concatenate([x,xv],axis=0)
L = len(reses)
for i in range(L):
    c = reses[i][0](xx)
    cs.append(c)
D = numpy.zeros([L,L])
for i in range(L):
    for j in range(L):
            D[i,j] = utils.linear_distance(cs[i],cs[j])

plt.imshow(1-D,aspect='auto',interpolation='none')
plt.colorbar()
plt.xticks(range(L),r_title)
plt.yticks(range(L),r_title)
plt.show()

#%%
########################################################
# 4. A GAE path
########################################################

alpha_list = numpy.arange(0,1.1,0.1)
result_file_name = 'result/temp/gae_analysis_result_gae.pkl'
LOADRESULT = True
if not LOADRESULT:
    gaeres = []
    for alpha in alpha_list:
        res = learner.AE_train(x,y,alpha=alpha,randseed=0)
        gaeres.append(res)
    utils.saveobj(result_file_name,gaeres)
else:
    gaeres = utils.loadobj(result_file_name)

allres = [pcares] + gaeres
alltit = ['PCA'] + map(lambda x: 'GAE-'+str(x),alpha_list)

recerrs = []
prederrs = []
for res in allres:
    enc,dec = res
    c = enc(xv)
    x_new = dec(c)
    recerrs.append(utils.mse(x_new,xv))
    ct = enc(x)
    enet = ElasticNetCV(random_state=0,cv=3)   
    enet.fit(ct,y.ravel())
    yp = enet.predict(c)
    prederrs.append(utils.mse(yp.ravel(),yv.ravel()))

plt.figure(figsize=[5,5])
for i in range(len(recerrs)):
    xa,ya = recerrs[i],prederrs[i]
    plt.scatter(xa,ya)
    plt.text(xa,ya,alltit[i])
plt.xlabel('reconstruction loss')
plt.ylabel('prediction loss')
plt.savefig('result/fig/gae_path_toy.pdf', bbox_inches='tight')


plt.figure(figsize=[12,6])
jlst = [0,1,2,4,6,8]
for j in range(len(jlst)):
    plt.subplot(2,3,j+1)
    i = jlst[j]
    res = allres[i]
    learner.visualizer(x,res[0],res[1],alltit[i],y)
#    c = allres[i][0](x)
#    plt.scatter(c[:,0],c[:,1],c=y,cmap='hot',
#                s=40,
#                edgecolors='none',vmax=3,vmin=-3)
#    plt.title(alltit[i])
plt.tight_layout()
plt.savefig('result/fig/gae_visual_toy.pdf', bbox_inches='tight')
