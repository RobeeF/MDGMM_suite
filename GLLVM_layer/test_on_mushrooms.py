# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 11:33:34 2020

@author: rfuchs
"""

import os 
os.chdir('C:/Users/rfuchs/Documents/GitHub/GLLVM_layer')

import pandas as pd
import autograd.numpy as np

from gower import gower_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import confusion_matrix

from copy import deepcopy

from glmlvm import glmlvm
from init_params import init_params, dim_reduce_init
from utils import misc, gen_categ_as_bin_dataset, \
        ordinal_encoding, compute_nj

from autograd.numpy.linalg import LinAlgError


###############################################################################################
##############        Clustering on the Mushrooms dataset (UCI)          ######################
###############################################################################################


# Importing and selecting data
mush = pd.read_csv('mushrooms/agaricus-lepiota.csv', sep = ',', header = None)
mush = mush.infer_objects()

y = mush.iloc[:,1:]

# Keep only the variables that have at least 2 modalities
one_modality_vars = np.array([len(set(y[col])) == 1 for col in y.columns])
y = y.iloc[:, ~one_modality_vars]

le = LabelEncoder()
labels = mush.iloc[:,0]
labels_oh = le.fit_transform(labels)

#Delete missing data
missing_idx = y.iloc[:, 10] != '?'
y = y[missing_idx]

labels = labels[missing_idx]
labels_oh = labels_oh[missing_idx]
k = len(np.unique(labels_oh))

#===========================================#
# Formating the data
#===========================================#

var_distrib = np.array(['categorical', 'categorical', 'categorical', 'bernoulli', 'categorical',\
                        'bernoulli', 'bernoulli', 'bernoulli', 'categorical', 'bernoulli',\
                        'categorical', 'categorical', 'categorical', 'categorical', 'categorical', \
                        'bernoulli', 'ordinal', 'categorical', 'categorical', \
                        'ordinal', 'categorical'])

ord_idx = np.where(var_distrib == 'ordinal')[0]

# Extract labels for each y_j and then perform dirty manual reordering
all_labels = [np.unique(y.iloc[:,idx]) for idx in ord_idx]
all_codes = [list(range(len(lab))) for lab in all_labels]    

# Encode ordinal data
for i, idx in enumerate(ord_idx):
    y.iloc[:,idx] = ordinal_encoding(y.iloc[:,idx], all_labels[i], all_codes[i])

y_categ_non_enc = deepcopy(y)
vd_categ_non_enc = deepcopy(var_distrib)

# Encode categorical datas
y, var_distrib = gen_categ_as_bin_dataset(y, var_distrib)

# Encode binary data
le = LabelEncoder()
for colname in y.columns:
    if y[colname].dtype != np.int64:
        y[colname] = le.fit_transform(y[colname])

nj, nj_bin, nj_ord = compute_nj(y, var_distrib)
y_np = y.values

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

# Launching the algorithm
r = 2
numobs = len(y)
M = r * 4
k = 2

seed = 1
init_seed = 2
    
eps = 1E-05
it = 30
maxstep = 100

# Prince init
prince_init = dim_reduce_init(y, k, r, nj, var_distrib, dim_red_method = 'prince', seed = None)
out = glmlvm(y_np, r, k, prince_init, var_distrib, nj, M, seed)
m, pred = misc(labels_oh, out['classes'], True) 
print(m)
print(confusion_matrix(labels_oh, pred))

# Random init
random_init = init_params(r, nj_bin, nj_ord, k, None)
out = glmlvm(y_np, r, k, random_init, var_distrib, nj, M, it, eps, maxstep, seed)
m, pred = misc(labels_oh, out['classes'], True) 
print(m)
print(confusion_matrix(labels_oh, pred))


#=======================================================================
# Performance measure : Finding the best specification for init and DDGMM
#=======================================================================

res_folder = 'C:/Users/rfuchs/Documents/These/Experiences/mixed_algos/mushrooms'

# GLMLVM
# Best one: r = 1 for prince and for random
numobs = len(y)
k = 2
    
eps = 1E-05
it = 30
maxstep = 100

nb_trials= 30
glmlvm_res = pd.DataFrame(columns = ['it_id', 'r', 'init' , 'micro', \
                                     'macro', 'silhouette'])
inits = ['random', 'MCA']

for r in range(2, 6):
    print(r)
    M = r * 2
    for init_alg in inits:
        for i in range(nb_trials):
            if init_alg == 'random':
                init = init_params(r, nj_bin, nj_ord, k, None)
            else:
                init = dim_reduce_init(y, k, r, nj, var_distrib,\
                                       dim_red_method = 'prince', seed = None)
        
            try:
                out = glmlvm(y_np, r, k, init, var_distrib, nj, M, it, eps, maxstep, seed = None)
                m, pred = misc(labels_oh, out['classes'], True) 
                
                try:
                    sil = silhouette_score(dm, pred, metric = 'precomputed') 
                except:
                    sil = np.nan
                    
                micro = precision_score(labels_oh, pred, average = 'micro')
                macro = precision_score(labels_oh, pred, average = 'macro')
                
                print('micro', micro)
                print('macro', macro)
                print('sil', sil)

                glmlvm_res = glmlvm_res.append({'it_id': i + 1, 'r': str(r),
                                                'init': init_alg, \
                                                'micro': micro, 'macro': macro, \
                                                'silhouette': sil}, ignore_index=True)
            except (ValueError, LinAlgError):
                glmlvm_res = glmlvm_res.append({'it_id': i + 1, 'r': str(r),\
                                                'init': init_alg, \
                                                'micro': np.nan, 'macro': np.nan, \
                                                'silhouette': np.nan}, ignore_index=True)
           
glmlvm_res.groupby(['r', 'init']).mean()
glmlvm_res.groupby(['r', 'init']).std()

# Store each init in a different file
glmlvm_res_mca = glmlvm_res[glmlvm_res['init'] == 'MCA']
glmlvm_res_random = glmlvm_res[glmlvm_res['init'] == 'random']

# Break down results by init
glmlvm_res_mca.groupby(['r']).mean().max()
glmlvm_res_mca.groupby(['r']).std()


glmlvm_res_random[['micro', 'macro']].isna().sum(axis = 0) 
glmlvm_res_random.groupby(['r']).mean().max()
glmlvm_res_random.groupby(['r']).std()

glmlvm_res_mca.to_csv(res_folder + '/glmlvm_res_mca.csv')
glmlvm_res_random.to_csv(res_folder + '/glmlvm_res_random.csv')


glmlvm_res = pd.read_csv(res_folder + '/glmlvm_res_best_sil.csv')
