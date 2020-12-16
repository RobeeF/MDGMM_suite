# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 11:33:34 2020

@author: rfuchs
"""

import os 
os.chdir('C:/Users/rfuchs/Documents/GitHub/MDGMM_suite/DDGMM')

from gower import gower_matrix

from init_params import dim_reduce_init
from ddgmm import DDGMM
from data_preprocessing import gen_categ_as_bin_dataset, \
        ordinal_encoding, compute_nj
from metrics import misc, cluster_purity

import pandas as pd
import autograd.numpy as np

from copy import deepcopy
from sklearn.metrics import precision_score
from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import silhouette_score


###############################################################################################
##############        Clustering on the Mushrooms dataset (UCI)          ######################
###############################################################################################

#===========================================#
# Importing and droping NaNs
#===========================================#
os.chdir('C:/Users/rfuchs/Documents/These/Stats/mixed_dgmm/datasets')

# Importing and selecting data
mush = pd.read_csv('mushrooms/agaricus-lepiota.csv', sep = ',', header = None)
mush = mush.infer_objects()

y = mush.iloc[:,1:]

# Keep only the variables that have at least 2 modalities
one_modality_vars = np.array([len(set(y[col])) == 1 for col in y.columns])
y = y.iloc[:, ~one_modality_vars]

# Encode the outcome
le = LabelEncoder()
labels = mush.iloc[:,0]
labels_oh = le.fit_transform(labels)

#Delete missing data
missing_idx = y.iloc[:, 10] != '?'
y = y[missing_idx]

labels = labels[missing_idx]
labels_oh = labels_oh[missing_idx]
n_clusters = len(np.unique(labels_oh))

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
#y, var_distrib = gen_categ_as_bin_dataset(y, var_distrib)

#######################################################
# Test to encode categorical variables
le = LabelEncoder()
for col_idx, colname in enumerate(y.columns):
    if var_distrib[col_idx] == 'categorical': 
        y[colname] = le.fit_transform(y[colname])
#################################################

# Encode binary data
le = LabelEncoder()
for col_idx, colname in enumerate(y.columns):
    if var_distrib[col_idx] == 'bernoulli': # Attention
        y[colname] = le.fit_transform(y[colname])

nj, nj_bin, nj_ord, nj_categ = compute_nj(y, var_distrib)
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

r = np.array([2, 1])
numobs = len(y)
k = [n_clusters]

seed = 1
init_seed = 2
    
eps = 1E-05
it = 30
maxstep = 100

# Prince init
prince_init = dim_reduce_init(y, n_clusters, k, r, nj, var_distrib, seed = None)


out = DDGMM(y_np, n_clusters, r, k, prince_init, var_distrib, nj, it, eps, maxstep, seed)
m, pred = misc(labels_oh, out['classes'], True) 
print(m)
print(confusion_matrix(labels_oh, pred))


#=======================================================================
# Performance measure : Finding the best specification for init and DDGMM
#=======================================================================

res_folder = 'C:/Users/rfuchs/Documents/These/Experiences/mixed_algos/mushrooms'

# Init
# Best one r = (3,1) and (2,1)
numobs = len(y)
k = [n_clusters]

nb_trials= 30
mca_res = pd.DataFrame(columns = ['it_id', 'r', 'micro', 'macro', 'silhouette'])

for r1 in range(2, 9):
    print(r1)
    r = np.array([r1, 1])
    for i in range(nb_trials):
        # Prince init
        try:
            prince_init = dim_reduce_init(y, n_clusters, k, r, nj, var_distrib, seed = None)
            m, pred = misc(labels_oh, prince_init['classes'], True) 
    
            sil = silhouette_score(dm, pred, metric = 'precomputed')        
            micro = precision_score(labels_oh, pred, average = 'micro')
            macro = precision_score(labels_oh, pred, average = 'macro')
    
            mca_res = mca_res.append({'it_id': i + 1, 'r': str(r), 'micro': micro, 'macro': macro, \
                                            'silhouette': sil}, ignore_index=True)
        except:
            mca_res = mca_res.append({'it_id': i + 1, 'r': str(r), 'micro': np.nan, 'macro': np.nan, \
                                            'silhouette': np.nan}, ignore_index=True)           
        
mca_res.groupby('r').mean()
mca_res.groupby('r').std()

mca_res.to_csv(res_folder + '/mca_res.csv')

#==================================================
# DDGMM First fing the best architecture
#==================================================

# DDGMM. Thresholds use: 0.25 and 0.10
# r = [5, 1]
# k = [2]
r = np.array([5, 4, 3])
numobs = len(y)
k = [4, n_clusters]
eps = 1E-05
it = 3
maxstep = 100
seed = None

 
prince_init = dim_reduce_init(y, n_clusters, k, r, nj, var_distrib, seed = None)
out = DDGMM(y_np, n_clusters, r, k, prince_init, var_distrib, nj, it, eps, maxstep, seed = None)

r = out['best_r']
numobs = len(y)
k = out['best_k']

it = 30
nb_trials= 30
ddgmm_res = pd.DataFrame(columns = ['it_id', 'micro', 'macro', 'silhouette'])

for i in range(nb_trials):

    print(i)
    # Prince init
    prince_init = dim_reduce_init(y, n_clusters, k, r, nj, var_distrib, seed = None)

    try:
        out = DDGMM(y_np, n_clusters, r, k, prince_init, var_distrib, nj, it,\
            eps, maxstep, seed, perform_selec = False)
        m, pred = misc(labels_oh, out['classes'], True) 

        sil = silhouette_score(dm, pred, metric = 'precomputed')                
        micro = precision_score(labels_oh, pred, average = 'micro')
        macro = precision_score(labels_oh, pred, average = 'macro')

        ddgmm_res = ddgmm_res.append({'it_id': i + 1, 'micro': micro, 'macro': macro, \
                                    'silhouette': sil}, ignore_index=True)
    except:
        ddgmm_res = ddgmm_res.append({'it_id': i + 1, 'micro': np.nan, 'macro': np.nan, \
                                    'silhouette': np.nan}, ignore_index=True)

    print(sil)
    print(micro)
    print(macro)
    print(ddgmm_res)
    ddgmm_res.to_csv(res_folder + '/ddgmm_res_categ_encoded_best_sil_identif2.csv')


ddgmm_res.mean()
ddgmm_res.std()

ddgmm_res.to_csv(res_folder + '/ddgmm_res.csv')
ddgmm_res = pd.read_csv(res_folder + '/ddgmm_res_categ_encoded.csv')


#=======================================================================
# Performance measure : Finding the best specification for other algos
#=======================================================================

from gower import gower_matrix
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes
from sklearn.cluster import AgglomerativeClustering
from minisom import MiniSom   
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


res_folder = 'C:/Users/rfuchs/Documents/These/Experiences/mixed_algos/mushrooms'

# <nb_trials> tries for each specification
nb_trials = 30


#****************************
# Partitional algorithm
#****************************

part_res_modes = pd.DataFrame(columns = ['it_id', 'init', 'micro', 'macro', 'silhouette'])

inits = ['Huang', 'Cao', 'random']

for init in inits:
    print(init)
    for i in range(nb_trials):
        km = KModes(n_clusters= n_clusters, init=init, n_init=10, verbose=0)
        kmo_labels = km.fit_predict(y_np_nenc)
        m, pred = misc(labels_oh, kmo_labels, True) 
        
        sil = silhouette_score(dm, pred, metric = 'precomputed')                        
        micro = precision_score(labels_oh, pred, average = 'micro')
        macro = precision_score(labels_oh, pred, average = 'macro')

        part_res_modes = part_res_modes.append({'it_id': i + 1, 'init': init, \
                            'micro': micro, 'macro': macro, 'silhouette': sil}, \
                                               ignore_index=True)
            
# Cao best spe
part_res_modes.groupby('init').mean().max()
part_res_modes.groupby('init').std() 

part_res_modes.to_csv(res_folder + '/part_res_modes_categ_encoded.csv')

#****************************
# K prototypes
#****************************

part_res_proto = pd.DataFrame(columns = ['it_id', 'init', 'micro', 'macro', 'silhouette'])


for init in inits:
    print(init)
    for i in range(nb_trials):
        km = KPrototypes(n_clusters = n_clusters, init = init, n_init=10, verbose=0)
        kmo_labels = km.fit_predict(y_np_nenc, categorical = np.where(cf_non_enc)[0].tolist())
        m, pred = misc(labels_oh, kmo_labels, True) 
        
        sil = silhouette_score(dm, pred, metric = 'precomputed')                                
        micro = precision_score(labels_oh, pred, average = 'micro')
        macro = precision_score(labels_oh, pred, average = 'macro')

        part_res_proto = part_res_proto.append({'it_id': i + 1, 'init': init, \
                            'micro': micro, 'macro': macro, 'silhouette': sil}, \
                                               ignore_index=True)

# Random is best
part_res_proto.groupby('init').mean().max()
part_res_proto.groupby('init').std()

part_res_proto.to_csv(res_folder + '/part_res_proto_categ_encoded.csv')

#****************************
# Hierarchical clustering
#****************************

hierarch_res = pd.DataFrame(columns = ['it_id', 'linkage', 'micro', 'macro', 'silhouette'])

linkages = ['complete', 'average', 'single']

for linky in linkages: 
    for i in range(nb_trials):  
        aglo = AgglomerativeClustering(n_clusters = n_clusters, affinity ='precomputed', linkage = linky)
        aglo_preds = aglo.fit_predict(dm)
        m, pred = misc(labels_oh, aglo_preds, True) 
        
        
        sil = silhouette_score(dm, pred, metric = 'precomputed')                        
        micro = precision_score(labels_oh, pred, average = 'micro')
        macro = precision_score(labels_oh, pred, average = 'macro')

        hierarch_res = hierarch_res.append({'it_id': i + 1, 'linkage': linky, \
                            'micro': micro, 'macro': macro, 'silhouette': sil},\
                                           ignore_index=True)

 
hierarch_res.groupby('linkage').mean()
hierarch_res.groupby('linkage').std()

hierarch_res.to_csv(res_folder + '/hierarch_res_categ_encoded.csv')

#****************************
# Neural-network based
#****************************

som_res = pd.DataFrame(columns = ['it_id', 'sigma', 'lr' ,'micro', 'macro', 'silhouette'])
y_np = y.values.astype(float)
numobs = len(y)

sigmas = np.linspace(0.001, 3, 5)
lrs = np.linspace(0.0001, 0.5, 10)

for sig in sigmas:
    for lr in lrs:
        for i in range(nb_trials):
            som = MiniSom(n_clusters, 1, y_np.shape[1], sigma = sig, learning_rate = lr) # initialization of 6x6 SOM
            som.train(y_np, 100) # trains the SOM with 100 iterations
            som_labels = [som.winner(y_np[i])[0] for i in range(numobs)]
            m, pred = misc(labels_oh, som_labels, True) 
            
            try:
                sil = silhouette_score(dm, pred, metric = 'precomputed')  
            except ValueError:
                sil = np.nan
                
            micro = precision_score(labels_oh, pred, average = 'micro')
            macro = precision_score(labels_oh, pred, average = 'macro')


            som_res = som_res.append({'it_id': i + 1, 'sigma': sig, 'lr': lr, \
                            'micro': micro, 'macro': macro, 'silhouette': sil},\
                                     ignore_index=True)
   
som_res.groupby(['sigma', 'lr']).mean()
som_res.groupby(['sigma', 'lr']).mean().max()

som_res.groupby(['sigma', 'lr']).std()
som_res.to_csv(res_folder + '/som_res_categ_encoded.csv')


#****************************
# Other algorithms family
#****************************

dbs_res = pd.DataFrame(columns = ['it_id', 'data' ,'leaf_size', 'eps',\
                                  'min_samples','micro', 'macro', 'silhouette'])

lf_size = np.arange(1,6) * 10
epss = np.linspace(0.01, 5, 5)
min_ss = np.arange(1, 5)
data_to_fit = ['scaled', 'gower']

for lfs in lf_size:
    print("Leaf size:", lfs)
    for eps in epss:
        for min_s in min_ss:
            for data in data_to_fit:
                for i in range(nb_trials):
                    if data == 'gower':
                        dbs = DBSCAN(eps = eps, min_samples = min_s, \
                                     metric = 'precomputed', leaf_size = lfs).fit(dm)
                    else:
                        dbs = DBSCAN(eps = eps, min_samples = min_s, leaf_size = lfs).fit(y_np)
                        
                    dbs_preds = dbs.labels_
                    
                    if len(np.unique(dbs_preds)) > n_clusters:
                        continue
                    
                    m, pred = misc(labels_oh, dbs_preds, True) 
                    
                    try:
                        sil = silhouette_score(dm, pred, metric = 'precomputed')     
                    except ValueError:     
                        sil = np.nan
                          
                    micro = precision_score(labels_oh, pred, average = 'micro')
                    macro = precision_score(labels_oh, pred, average = 'macro')

    
                    dbs_res = dbs_res.append({'it_id': i + 1, 'leaf_size': lfs, \
                                'eps': eps, 'min_samples': min_s, 'micro': micro,\
                                    'data': data, 'macro': macro, 'silhouette': sil},\
                                             ignore_index=True)

# scaled data eps = 3.7525 and min_samples = 4  is the best spe
mean_res = dbs_res.groupby(['data','leaf_size', 'eps', 'min_samples']).mean()
maxs = mean_res.max()
print(maxs)

dbs_res.set_index(['data','leaf_size', 'eps', 'min_samples'])[mean_res['micro'] == maxs['micro']].std()
dbs_res.set_index(['data','leaf_size', 'eps', 'min_samples'])[mean_res['macro'] == maxs['macro']].std()
dbs_res.set_index(['data','leaf_size', 'eps', 'min_samples'])[mean_res['silhouette'] == maxs['silhouette']].std()

dbs_res.to_csv(res_folder + '/dbs_res_categ_encoded.csv')
