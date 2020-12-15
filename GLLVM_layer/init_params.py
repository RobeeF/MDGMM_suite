# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 18:00:53 2020

@author: rfuchs
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 16:55:44 2020

@author: Utilisateur
"""

from utils import gen_categ_as_bin_dataset

from sklearn import manifold
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder 
from sklearn.linear_model import LogisticRegression

import umap
import prince
import pandas as pd

# Dirty local hard copy of the Github bevel package
from bevel.linear_ordinal_regression import  OrderedLogit 

import autograd.numpy as np
from autograd.numpy.random import uniform
from autograd.numpy import newaxis as n_axis
from autograd.numpy import transpose as t
from autograd.numpy.linalg import cholesky, pinv


####################################################################################
########################## Random initialisations ##################################
####################################################################################

def init_params(r, nj_bin, nj_ord, k, init_seed):
    ''' Generate random initialisations for the parameters
    
    r (int): The dimension of latent variables
    nj_bin (nb_bin 1darray): For binary/count data: The maximum values that the variable can take. 
    nj_ord (nb_ord 1darray): For ordinal data: the number of different existing categories for each variable
    k (int): The number of components of the latent Gaussian mixture
    init_seed (int): The random state seed to set (Only for numpy generated data for the moment)
    --------------------------------------------------------------------------------------------
    returns (dict): The initialisation parameters   
    '''
    
    # Seed for init
    np.random.seed = init_seed
    init = {}
    
    
    # Gaussian mixture params
    init['w'] = np.full(k, 1/k) 
    
    mu_init = np.repeat(np.linspace(-1.0, 1.0, num = k)[..., n_axis], axis = 1, repeats =r)
    init['mu'] = (uniform(low = -5, high = 5, size = (1,1)) * mu_init)
    init['mu'] = init['mu'][..., np.newaxis]
  
    init['sigma'] = np.zeros(shape = (k, r, r))
    for i in range(k):
        init['sigma'][i] = 0.050 * np.eye(r)
        
    # Enforcing identifiability constraints
    muTmu = init['mu'] @ t(init['mu'], (0,2,1))  
     
    E_zzT = (init['w'][..., n_axis, n_axis] * (init['sigma'] + muTmu)).sum(0, keepdims = True)
    Ezz_T = (init['w'][...,n_axis, n_axis] * init['mu']).sum(0, keepdims = True)
    
    var_z = E_zzT - Ezz_T @ t(Ezz_T, (0,2,1)) # Koenig-Huyghens Formula for Variance Computation
    sigma_z = cholesky(var_z)
     
    init['sigma'] = pinv(sigma_z) @ init['sigma'] @ t(pinv(sigma_z), (0, 2, 1))
    init['mu'] = pinv(sigma_z) @ init['mu']
    init['mu']  = init['mu']  - Ezz_T

    # GLLVM params    
    p1 = len(nj_bin)
    p2 = len(nj_ord)
    
    if p1 > 0:
        init['lambda_bin'] = uniform(low = -3, high = 3, size = (p1, r + 1))
        init['lambda_bin'][:,1:] = init['lambda_bin'][:,1:] @ sigma_z[0] 
        
        if (r > 1): 
            init['lambda_bin'] = np.tril(init['lambda_bin'], k = 1)

    else:
        init['lambda_bin'] = np.array([]) #np.full((p1, r + 1), np.nan)
  
    if p2 > 0:

        lambda_ord = []
        for j in range(p2):
            lambda0_ord = np.sort(uniform(low = -2, high = 2, size = (nj_ord[j] - 1)))
            Lambda_ord = uniform(low = -3, high = 3, size = r)
            lambda_ord.append(np.hstack([lambda0_ord, Lambda_ord]))
              
        init['lambda_ord'] = lambda_ord
        
    else:
        init['lambda_ord'] = np.array([])#np.full((p2, 1), np.nan)

    return(init)


####################################################################################
########## MCA / T-SNE / UMAP + GMM + Logistic Regressions initialisation ##########
####################################################################################

def bin_to_bern(Nj, yj_binom, zM_binom):
    ''' Split the binomial variable into Bernoulli. Them just recopy the corresponding zM.
    It is necessary to fit binary logistic regression
    Example: yj has support in [0,10]: Then if y_ij = 3 generate a vector with 3 ones and 7 zeros 
    (3 success among 10).
    
    Nj (int): The upper bound of the support of yj_binom
    yj_binom (numobs 1darray): The Binomial variable considered
    zM_binom (numobs x r nd-array): The continuous representation of the data
    -----------------------------------------------------------------------------------
    returns (tuple of 2 (numobs x Nj) arrays): The "Bernoullied" Binomial variable
    '''
    
    n_yk = len(yj_binom) # parameter k of the binomial
    
    # Generate Nj Bernoullis from each binomial and get a (numobsxNj, 1) table
    u = uniform(size =(n_yk,Nj))
    p = (yj_binom/Nj)[..., n_axis]
    yk_bern = (u > p).astype(int).flatten('A')#[..., n_axis] 
        
    return yk_bern, np.repeat(zM_binom, Nj, 0)


def dim_reduce_init(y, k, r, nj, var_distrib, dim_red_method = 'prince', seed = None):
    ''' Perform dimension reduction into a continuous r dimensional space and determine 
    the init coefficients in that space
    
    y (numobs x p ndarray): The observations containing categorical variables
    k (int): The number of components of the latent Gaussian mixture
    r (int): The dimension of latent variables
    nj (p 1darray): For binary/count data: The maximum values that the variable can take. 
                    For ordinal data: the number of different existing categories for each variable
    var_distrib (p 1darray): An array containing the types of the variables in y 
    dim_red_method (str): Choices are 'prince' for MCA, 'umap' of 'tsne'
    seed (None): The random state seed to use for the dimension reduction
    M (int): The number of MC points to compute     
    ---------------------------------------------------------------------------------------
    returns (dict): All initialisation parameters
    '''
    
    #==============================================================
    # Dimension reduction performed with the dim_red_method chosen
    #==============================================================
    
    if dim_red_method == 'umap':
        reducer = umap.UMAP(n_components = r, random_state = seed)
        z_emb = reducer.fit_transform(y)
        
        
    elif dim_red_method == 'tsne':
        tsne = manifold.TSNE(n_components = r, init='pca', random_state = seed)
        z_emb = tsne.fit_transform(y)
        
        
    elif dim_red_method == 'prince':
        
        if type(y) != pd.core.frame.DataFrame:
            raise TypeError('y should be a dataframe for prince')
        
        # Check input False due to the new pandas update
        mca = prince.MCA(n_components = r, n_iter=3, copy=True, \
                         check_input=False, engine='auto', random_state=42)
        mca = mca.fit(y)
        z_emb = mca.row_coordinates(y).values.astype(float)
        
        y = y.values.astype(int)
        
    elif dim_red_method == 'famd':
        famd = prince.FAMD(n_components = r, n_iter=3, copy=True, check_input=False, \
                               engine='auto', random_state = seed)
        z_emb = famd.fit_transform(y).values
            
        # Encode categorical datas
        y, var_distrib = gen_categ_as_bin_dataset(y, var_distrib)
        
        # Encode binary data
        le = LabelEncoder()
        for col_idx, colname in enumerate(y.columns):
            if var_distrib[col_idx] == 'bernoulli':
                y[colname] = le.fit_transform(y[colname])
                
        y = y.values#.astype(int)

    else:
        raise ValueError('Only tnse, umap and prince initialisation are available not ', dim_red_method)
    

    #==============================================================
    # Set the parameters of each data type
    #==============================================================    
    
    y_bin = y[:, np.logical_or(var_distrib == 'bernoulli',\
                               var_distrib == 'binomial')].astype(int)   
    nj_bin = nj[np.logical_or(var_distrib == 'bernoulli',\
                              var_distrib == 'binomial')]
    nb_bin = len(nj_bin)
    
    y_ord = y[:, var_distrib == 'ordinal'].astype(int)    
    nj_ord = nj[var_distrib == 'ordinal']
    nb_ord = len(nj_ord)
    
    #=======================================================
    # Determining the Gaussian Parameters
    #=======================================================
    init = {}

    gmm = GaussianMixture(n_components = k, covariance_type='full').fit(z_emb)
    mu = gmm.means_[..., n_axis]
    sigma = gmm.covariances_  
    w = gmm.weights_
    
    # Enforcing identifiability constraints
    muTmu = mu @ t(mu, (0,2,1))  
     
    E_zzT = (w[..., n_axis, n_axis] * (sigma + muTmu)).sum(0, keepdims = True)
    Ezz_T = (w[...,n_axis, n_axis] * mu).sum(0, keepdims = True)
    
    var_z = E_zzT - Ezz_T @ t(Ezz_T, (0,2,1)) # Koenig-Huyghens Formula for Variance Computation
    sigma_z = cholesky(var_z)
     
    sigma = pinv(sigma_z) @ sigma @ t(pinv(sigma_z), (0, 2, 1))
    init['sigma'] = sigma
    
    mu = pinv(sigma_z) @ mu
    init['mu']  = mu  - Ezz_T        
    
    init['w'] = w
    init['classes'] = gmm.predict(z_emb)
         
    #=======================================================
    # Determining the coefficients of the GLLVM layer
    #=======================================================
    
    # Determining lambda_bin coefficients.
    
    lambda_bin = np.zeros((nb_bin, r + 1))
    
    for j in range(nb_bin): 
        Nj = np.max(y_bin[:,j]) # The support of the jth binomial is [1, Nj]
        
        if Nj ==  1:  # If the variable is Bernoulli not binomial
            yj = y_bin[:,j]
            z = z_emb
        else: # If not, need to convert Binomial output to Bernoulli output
            yj, z = bin_to_bern(Nj, y_bin[:,j], z_emb)
        
        lr = LogisticRegression()
        
        if j < r - 1:
            lr.fit(z[:,:j + 1], yj)
            lambda_bin[j, :j + 2] = np.concatenate([lr.intercept_, lr.coef_[0]])
        else:
            lr.fit(z, yj)
            lambda_bin[j] = np.concatenate([lr.intercept_, lr.coef_[0]])
    
    ## Identifiability of bin coefficients
    lambda_bin[:,1:] = lambda_bin[:,1:] @ sigma_z[0] 
    
    # Determining lambda_ord coefficients
    lambda_ord = []
    
    for j in range(nb_ord):
        Nj = len(np.unique(y_ord[:,j], axis = 0))  # The support of the jth ordinal is [1, Nj]
        yj = y_ord[:,j]
        
        ol = OrderedLogit()
        ol.fit(z_emb, yj)
        
        ## Identifiability of ordinal coefficients
        beta_j = (ol.beta_.reshape(1, r) @ sigma_z[0]).flatten()
        lambda_ord_j = np.concatenate([ol.alpha_, beta_j])
        lambda_ord.append(lambda_ord_j)        
        
    init['lambda_bin'] = lambda_bin
    init['lambda_ord'] = lambda_ord
        
    return init


