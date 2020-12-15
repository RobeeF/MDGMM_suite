# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 08:52:28 2020

@author: Utilisateur
"""


from lik_functions import log_py_zM_bin, log_py_zM_ord, binom_loglik_j,\
        ord_loglik_j
from lik_gradients import bin_grad_j, ord_grad_j


from numeric_stability import ensure_psd

import autograd.numpy as np
from autograd.numpy import newaxis as n_axis
from autograd.numpy import transpose as t
from autograd.numpy.random import multivariate_normal
from autograd.numpy.linalg import cholesky, pinv

from scipy.linalg import block_diag
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import multivariate_normal as mvnorm
 
from copy import deepcopy
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint

import warnings
warnings.filterwarnings("default")

def glmlvm(y, r, k, init, var_distrib, nj, M, it = 50, eps = 1E-05, maxstep = 100, seed = None): 
    ''' Fit a Generalized Linear Mixture of Latent Variables Model (GLMLVM)
    
    y (numobs x p ndarray): The observations containing discrete variables
    r (int): The dimension of latent variables
    k (int): The number of components of the latent Gaussian mixture
    init (dict): The initial values of the parameters
    var_distrib (p 1darray): An array containing the types of the variables in y 
    nj (p 1darray): For binary/count data: The maximum values that the variable can take. 
                    For ordinal data: the number of different existing categories for each variable
    M (int): The number of MC points to compute 
    it (int): The maximum number of EM iterations of the algorithm
    eps (float): If the likelihood increase by less than eps then the algorithm stops
    maxstep (int): The maximum number of optimisation step for each variable
    seed (int): The random state seed to set (Only for numpy generated data for the moment)
    ------------------------------------------------------------------------------------------------
    returns (dict): The predicted classes and the likelihood through the EM steps
    '''

    epsilon = 1E-16
    prev_lik = - 1E16
    tol = 0.01
    
    # Initialize the parameters
    mu = deepcopy(init['mu'])
    sigma = deepcopy(init['sigma'])
    lambda_bin = deepcopy(init['lambda_bin'])
    lambda_ord = deepcopy(init['lambda_ord'])

    w = deepcopy(init['w'])
    
    numobs = len(y)
    likelihood = []
    hh = 0
    ratio = 1000
    classes = np.zeros((numobs))
    np.random.seed = seed
        
    # Dispatch variables between categories
    y_bin = y[:, np.logical_or(var_distrib == 'bernoulli',var_distrib == 'binomial')]
    nj_bin = nj[np.logical_or(var_distrib == 'bernoulli',var_distrib == 'binomial')]
    nb_bin = len(nj_bin)
        
    y_ord = y[:, var_distrib == 'ordinal']    
    nj_ord = nj[var_distrib == 'ordinal'].astype(int)
    nb_ord = len(nj_ord)
        
    assert nb_ord + nb_bin  > 0 
                     
    while ((hh < it) & (ratio > eps)):
        hh = hh + 1
        log_py_zM = np.zeros((M, numobs, k))

        #=================================================
        # Simulate pseudo-observations
        #=================================================
        
        zM = multivariate_normal(size = (M, 1), mean = mu.flatten(order = 'F'), cov = block_diag(*sigma)) 
        zM = t(zM.reshape(M, k, r, order = 'F'), (0, 2, 1))
        
        #==================================================
        # Compute the p(y| zM) for all variable categories
        #==================================================
        
        if nb_bin: # First the Count/Binomial variables
            log_py_zM = log_py_zM + log_py_zM_bin(lambda_bin, y_bin, zM, k, nj_bin) 
                
        if nb_ord: # Then the ordinal variables 
            log_py_zM = log_py_zM + log_py_zM_ord(lambda_ord, y_ord, zM, k, nj_ord)[:,:,:,0] 
        
        py_zM = np.exp(log_py_zM)
        py_zM = np.where(py_zM == 0, 1E-50, py_zM)

        #####################################################################################
        ################################# E step ############################################
        #####################################################################################
        
        #======================================================
        # Resample zM conditionally on y and s
        #======================================================
        
        qM = py_zM / np.sum(py_zM, axis = 0, keepdims = True)
        new_zM = np.zeros((M,numobs, r, k))
        
        new_zM = np.zeros((M, numobs, r, k))
        for i in range(k):
            qM_cum = qM[:,:, i].T.cumsum(axis=1)
            u = np.random.rand(numobs, 1, M)
            
            choices = u < qM_cum[..., np.newaxis]
            idx = choices.argmax(1)
            
            new_zM[:,:,:,i] = np.take(zM[:,:, i], idx.T, axis=0)
        
        del(u)
        
        
        #=======================================================
        # Compute conditional probabilities used in the appendix
        #=======================================================
        
        pz_s = np.zeros((M, 1, k))
                
        for i in range(k): # Have to retake the function for DGMM to parallelize or use apply along axis
            pz_s[:,:, i] = mvnorm.pdf(zM[:,:,i], mean = mu[i].flatten(), cov = sigma[i])[..., n_axis]
                
        # Compute (17) p(y | s_i = 1)
        norm_cste = np.sum(pz_s, axis = 0, keepdims = True)
        norm_cste = np.where(norm_cste == 0.0, epsilon, norm_cste)
        pz_s_norm = pz_s / norm_cste
        py_s = (pz_s_norm * py_zM).sum(axis = 0)
        
        # Compute (16) p(z |y, s) 
        norm_cste = py_s[n_axis]
        norm_cste = np.where(norm_cste == 0.0, epsilon, norm_cste)      
        p_z_ys = pz_s * py_zM / norm_cste
        
        norm_cste = np.sum(p_z_ys, axis = 0, keepdims = True)
        norm_cste = np.where(norm_cste == 0.0, epsilon, norm_cste)           
        p_z_ys = p_z_ys / norm_cste # Normalizing p(z|y,s)
        
        # Free some memory
        del(py_zM)
        del(pz_s_norm)
        del(pz_s)
        del(qM)
        
        # Compute unormalized (18)
        ps_y = w[n_axis] * py_s
    
        norm_cste = np.sum(ps_y, axis = 1, keepdims = True)   
        norm_cste = np.where(norm_cste == 0.0, epsilon, norm_cste)           
        ps_y = ps_y / norm_cste  
        p_y = py_s @ w
        
        # Compute E_{y,s}(z) and E_{y,s}(zTz)
        E_z_sy = t(np.mean(new_zM, axis = 0), (0, 2, 1)) 
        zTz = (t(new_zM[...,n_axis], (0, 1, 3, 2, 4)) @ \
                   t(new_zM[...,n_axis], (0, 1, 3, 4, 2)))
        E_zz_sy = np.mean(zTz, axis = 0)
        
        # Compute E_y(z) might be useful for ploting purposes
        Ez_y = (ps_y[...,n_axis] * E_z_sy).sum(1)
                
        del(new_zM)
        
        #=======================================================
        # Compute Gaussian Parameters
        #=======================================================

        w = np.mean(ps_y, axis = 0)
        den = ps_y.sum(0, keepdims = True).T[..., n_axis]
        den = np.where(den == 0.0, epsilon, den)
        
        mu = (ps_y[...,n_axis] * E_z_sy).sum(0)[..., np.newaxis] / den

        muTmu = mu @ t(mu, (0,2,1))  
        sigma = np.sum(ps_y[..., n_axis, n_axis] * (E_zz_sy - \
                    muTmu[n_axis]), axis = 0) / den

        sigma = ensure_psd(sigma)
         
        # Enforcing identifiability constraints
        E_zzT = (w[..., n_axis, n_axis] * (sigma + muTmu)).sum(0, keepdims = True)
        Ezz_T = (w[...,n_axis, n_axis] * mu).sum(0, keepdims = True)

        var_z = E_zzT - Ezz_T @ t(Ezz_T, (0,2,1)) # Koenig-Huyghens Formula for Variance Computation
        sigma_z = cholesky(var_z)
        
        sigma = pinv(sigma_z) @ sigma @ t(pinv(sigma_z), (0, 2, 1))
        mu = pinv(sigma_z) @ mu
        mu  = mu  - Ezz_T
        
        del(E_z_sy)
        del(E_zz_sy)

        sigma = ensure_psd(sigma)

        
        ###########################################################################
        ############################ M step #######################################
        ###########################################################################
        
        # We optimize each column separately as it is faster than all column jointly 
        # (and more relevant with the independence hypothesis)
        
        #=======================================================
        # Binomial link parameters
        #=======================================================        
        
        for j in range(nb_bin):
            if j < r - 1: # Constrained columns
                nb_constraints = r - j - 1
                lcs = np.hstack([np.zeros((nb_constraints, j + 2)), np.eye(nb_constraints)])
                linear_constraint = LinearConstraint(lcs, np.full(nb_constraints, 0), \
                                                 np.full(nb_constraints, 0), keep_feasible = True)
            
                opt = minimize(binom_loglik_j, lambda_bin[j] , args = (y_bin[:,j], zM, k, ps_y, p_z_ys, nj_bin[j]), 
                               tol = tol, method='trust-constr',  jac = bin_grad_j, \
                               constraints = linear_constraint, hess = '2-point', options = {'maxiter': maxstep})
                        
            else: # Unconstrained columns
                opt = minimize(binom_loglik_j, lambda_bin[j], args = (y_bin[:,j], zM, k, ps_y, p_z_ys, nj_bin[j]), 
                               tol = tol, method='BFGS', jac = bin_grad_j, options = {'maxiter': maxstep})
            
            res = opt.x
            if not(opt.success):
                res = lambda_bin[j]
                warnings.warn('One of the binomial optimisations has failed', RuntimeWarning)
                #raise RuntimeError('Binomial optimization failed')
                
            lambda_bin[j, :] = deepcopy(res)  

        # Last identifiability part
        if nb_bin > 0:
            lambda_bin[:,1:] = lambda_bin[:,1:] @ sigma_z[0] 
  
        
        #=======================================================
        # Ordinal link parameters
        #=======================================================    
        
        for j in range(nb_ord):
            enc = OneHotEncoder(categories='auto')
            y_oh = enc.fit_transform(y_ord[:,j][..., n_axis]).toarray()                
            
            # Define the constraints such that the threshold coefficients are ordered
            nb_constraints = nj_ord[j] - 2 
            nb_params = nj_ord[j] + r - 1
            
            lcs = np.full(nb_constraints, -1)
            lcs = np.diag(lcs, 1)
            np.fill_diagonal(lcs, 1)
            
            lcs = np.hstack([lcs[:nb_constraints, :], np.zeros([nb_constraints, nb_params - (nb_constraints + 1)])])
            
            linear_constraint = LinearConstraint(lcs, np.full(nb_constraints, -np.inf), \
                                                 np.full(nb_constraints, 0), keep_feasible = True)
            
            warnings.filterwarnings("default")
        
            opt = minimize(ord_loglik_j, lambda_ord[j] , args = (y_oh, zM, k, ps_y, p_z_ys, nj_ord[j]), 
                               tol = tol, method='trust-constr',  jac = ord_grad_j, \
                               constraints = linear_constraint, hess = '2-point', options = {'maxiter': maxstep})
            
            res = opt.x
            if not(opt.success):
                res = lambda_ord[j]
                warnings.warn('One of the ordinal optimisations has failed', RuntimeWarning)
                #raise RuntimeError('Ordinal optimization failed')
                     
            # Ensure identifiability for Lambda_j
            new_lambda_ord_j = (res[-r: ].reshape(1, r) @ sigma_z[0]).flatten() 
            new_lambda_ord_j = np.hstack([deepcopy(res[: nj_ord[j] - 1]), new_lambda_ord_j]) # Complete with lambda_0 coefficients
            lambda_ord[j] = new_lambda_ord_j


        ###########################################################################
        ################## Clustering parameters updating #########################
        ###########################################################################
          
        new_lik = np.sum(np.log(p_y))
        likelihood.append(new_lik)
        ratio = (new_lik - prev_lik)/abs(prev_lik)
        
        if (hh < 3): 
            ratio = 2 * eps
        print(likelihood)
        
        # Refresh the classes only if they provide a better explanation of the data
        if prev_lik > new_lik:
            classes = np.argmax(ps_y, axis = 1) 
            
        prev_lik = new_lik


    out = dict(likelihood = likelihood, classes = classes)
    return(out)
