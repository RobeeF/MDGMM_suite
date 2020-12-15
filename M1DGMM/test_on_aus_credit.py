# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 13:25:11 2020

@author: rfuchs
"""

import os 

os.chdir('C:/Users/rfuchs/Documents/GitHub/M1DGMM')

from copy import deepcopy

from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import OneHotEncoder

from gower import gower_matrix
from sklearn.metrics import silhouette_score


import pandas as pd

from m1dgmm import M1DGMM
from init_params import dim_reduce_init
from metrics import misc, cluster_purity
from data_preprocessing import gen_categ_as_bin_dataset, \
        compute_nj

import autograd.numpy as np
from numpy.random import uniform


###############################################################################
######################## Credit data vizualisation    #########################
###############################################################################

#===========================================#
# Importing data
#===========================================#
os.chdir('C:/Users/rfuchs/Documents/These/Stats/mixed_dgmm/datasets')

credit = pd.read_csv('australian_credit/australian.csv', sep = ' ', header = None)
y = credit.iloc[:,:-1]
labels = credit.iloc[:,-1]

y = y.infer_objects()
numobs = len(y)


n_clusters = len(np.unique(labels))
p = y.shape[1]

#===========================================#
# Formating the data
#===========================================#
var_distrib = np.array(['bernoulli', 'continuous', 'continuous', 'categorical',\
                        'categorical', 'categorical', 'continuous', 'bernoulli',\
                        'bernoulli', 'continuous', 'bernoulli', 'categorical',\
                        'continuous', 'continuous']) 
 
# No ordinal data 
 
y_categ_non_enc = deepcopy(y)
vd_categ_non_enc = deepcopy(var_distrib)

# Encode categorical datas
le = LabelEncoder()
for col_idx, colname in enumerate(y.columns):
    if var_distrib[col_idx] == 'categorical': 
        y[colname] = le.fit_transform(y[colname])

# No binary data 

enc = OneHotEncoder(sparse = False, drop = 'first')
labels_oh = enc.fit_transform(np.array(labels).reshape(-1,1)).flatten()

nj, nj_bin, nj_ord, nj_categ = compute_nj(y, var_distrib)
y_np = y.values
nb_cont = np.sum(var_distrib == 'continuous')

p_new = y.shape[1]

# Feature category (cf)
cf_non_enc = np.logical_or(vd_categ_non_enc == 'categorical', vd_categ_non_enc == 'bernoulli')

# Non encoded version of the dataset:
y_nenc_typed = y_categ_non_enc.astype(np.object)
y_np_nenc = y_nenc_typed.values

# Defining distances over the non encoded features
dm = gower_matrix(y_nenc_typed, cat_features = cf_non_enc) 

dtype = {y.columns[j]: np.float64 if (var_distrib[j] != 'bernoulli') and \
        (var_distrib[j] != 'categorical') else np.str for j in range(p_new)}

y = y.astype(dtype, copy=True)


#===========================================#
# Running the algorithm
#===========================================# 

r = np.array([2, 1])
numobs = len(y)
k = [n_clusters]

seed = 1
init_seed = 2
    
eps = 1E-05
it = 20
maxstep = 100


prince_init = dim_reduce_init(y, n_clusters, k, r, nj, var_distrib, seed = None,\
                              use_famd=True)
m, pred = misc(labels_oh, prince_init['classes'], True) 
print(m)
print(confusion_matrix(labels_oh, pred))
print(silhouette_score(dm, pred, metric = 'precomputed'))


out = M1DGMM(y_np, n_clusters, r, k, prince_init, var_distrib, nj, it, eps, maxstep, seed)
m, pred = misc(labels_oh, out['classes'], True) 
print(m)
print(confusion_matrix(labels_oh, pred))
print(silhouette_score(dm, pred, metric = 'precomputed'))

# Plot the final groups

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

colors = ['green','red']

fig = plt.figure(figsize=(8,8))
plt.scatter(out["z"][:, 0], out["z"][:, 1], c=pred,\
            cmap=matplotlib.colors.ListedColormap(colors))

cb = plt.colorbar()
cb.ax.get_yaxis().set_ticks([])
for j, lab in enumerate(['0','1']):
    cb.ax.text(.5, (2 * j + 1) / 4.0, lab, ha='center', va='center', rotation=90)
cb.ax.get_yaxis().labelpad = 15


#=========================================================================
# Performance measure : Finding the best specification for init and DDGMM
#=========================================================================

res_folder = 'C:/Users/rfuchs/Documents/These/Experiences/mixed_algos/aus_credit'


# Init
# Best one r = (2,1)
numobs = len(y)
k = [n_clusters]

nb_trials= 30
mca_res = pd.DataFrame(columns = ['it_id', 'r', 'micro', 'macro', 'purity'])

for r1 in range(2, 9):
    print(r1)
    r = np.array([r1, 1])
    for i in range(nb_trials):
        # Prince init
        prince_init = dim_reduce_init(y, n_clusters, k, r, nj, var_distrib, seed = None)
        m, pred = misc(labels_oh, prince_init['classes'], True) 
        cm = confusion_matrix(labels_oh, pred)
        purity = cluster_purity(cm)
            
        micro = precision_score(labels_oh, pred, average = 'micro')
        macro = precision_score(labels_oh, pred, average = 'macro')
        #print(micro)
        #print(macro)
    
        mca_res = mca_res.append({'it_id': i + 1, 'r': str(r), 'micro': micro, 'macro': macro, \
                                        'purity': purity}, ignore_index=True)
       

mca_res.groupby('r').mean()
mca_res.groupby('r').std()

mca_res.to_csv(res_folder + '/mca_res.csv')

# MDGMM. Thresholds use: 0.25 and 0.10
# r = [5, 3, 2]
# k = [3, 2]
r = np.array([5, 4, 3])
numobs = len(y)
k = [4, n_clusters]
eps = 1E-05
it = 2
maxstep = 100

prince_init = dim_reduce_init(y, n_clusters, k, r, nj, var_distrib, \
                              seed = None, use_famd = True)
out = M1DGMM(y_np, n_clusters, r, k, prince_init, var_distrib, nj, it, eps,\
             maxstep, seed = None)

r = out['best_r']
numobs = len(y)
k = out['best_k']
eps = 1E-05
it = 30
maxstep = 100

nb_trials= 30
m1dgmm_res = pd.DataFrame(columns = ['it_id', 'micro', 'macro', 'silhouette'])

for i in range(nb_trials):

    print(i)
    # Prince init
    prince_init = dim_reduce_init(y, n_clusters, k, r, nj, var_distrib,\
                                  seed = None, use_famd = True)

    try:
        out = M1DGMM(y_np, n_clusters, r, k, prince_init, var_distrib, nj, it,\
                     eps, maxstep, seed = None, perform_selec = False)
        m, pred = misc(labels_oh, out['classes'], True) 

        sil = silhouette_score(dm, pred, metric = 'precomputed')
        micro = precision_score(labels_oh, pred, average = 'micro')
        macro = precision_score(labels_oh, pred, average = 'macro')
        print(micro)
        print(macro)

        m1dgmm_res = m1dgmm_res.append({'it_id': i + 1, 'micro': micro, 'macro': macro, \
                                    'silhouette': sil}, ignore_index=True)
    except:
        m1dgmm_res = m1dgmm_res.append({'it_id': i + 1, 'micro': np.nan, 'macro': np.nan, \
                                    'silhouette': np.nan}, ignore_index=True)



m1dgmm_res.mean()
m1dgmm_res.std()

m1dgmm_res.to_csv(res_folder + '/m1dgmm_res_famd.csv')
