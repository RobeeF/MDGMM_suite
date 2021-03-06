# -*- coding: utf-8 -*-
"""
Created on Thu May 14 18:26:18 2020

@author: RobF
"""

from scipy.linalg import block_diag
from scipy.stats import multivariate_normal as mvnorm
from copy import deepcopy

import autograd.numpy as np 
from autograd.numpy import transpose as t
from autograd.numpy import newaxis as n_axis
from autograd.numpy.random import multivariate_normal
from autograd.numpy.linalg import pinv

#=============================================================================
# MC Step functions
#=============================================================================

def draw_z_s(mu_s, sigma_s, eta, M, center_last_layer = False):
    ''' Draw from f(z^{l} | s) for all s in Omega and return the centered and
        non-centered draws
        
    mu_s (list of nd-arrays): The means of the Gaussians starting at each layer
    sigma_s (list of nd-arrays): The covariance matrices of the Gaussians 
                                                        starting at each layer
    eta (list of nb_layers elements of shape (K_l x r_{l-1}, 1)): mu parameters
                                                        for each layer
    M (dict of int): The number of MC to draw on each layer
    center_last_layer (Bool): Whether or not to return the last centered 
    -------------------------------------------------------------------------
    returns (list of ndarrays): z^{l} | s for all s in Omega and all l in L
    '''
    
    L = len(mu_s) 
    r = [mu_s[l].shape[1] for l in range(L)]
    S = [mu_s[l].shape[0] for l in range(L)]
    
    z_s = []
    zc_s = [] # z centered (denoted c) or all l

    for l in range(L): 
        zl_s = multivariate_normal(size = (M[l], 1), \
            mean = mu_s[l].flatten(order = 'C'), cov = block_diag(*sigma_s[l]))
            
        zl_s = zl_s.reshape(M[l], S[l], r[l], order = 'C')
        z_s.append(t(zl_s, (0, 2, 1)))

        if (l < L - 1) or center_last_layer:
            kl = eta[l].shape[0]
            eta_ = np.repeat(t(eta[l], (2, 0, 1)), S[l] // kl, axis = 1)
            zc_s.append(zl_s - eta_)
        
    return z_s, zc_s


def draw_z2_z1s(chsi, rho, M, r):
    ''' Draw from f(z^{l+1} | z^{l}, s, Theta) 
    chsi (list of nd-arrays): The chsi parameters for all paths starting at each layer
    rho (list of ndarrays): The rho parameters (covariance matrices) for
                                    all paths starting at each layer
    M (dict): The number of MC to draw on each layer
    r (dict): The dimension of each layer
    ---------------------------------------------------------------------------
    returns (list of nd-arrays): z^{l+1} | z^{l}, s, Theta for all (l,s)
    '''
    
    L = len(chsi)    
    S = [chsi[l].shape[0] for l in range(L)]
    
    z2_z1s = []
    for l in range(L):
        z2_z1s_l = np.zeros((M[l], M[l + 1], S[l], r[l + 1]))    
        for s in range(S[l]):
            for m in range(M[l]):
                z2_z1s_l[m,:, s] = multivariate_normal(size = M[l + 1], \
                    mean = rho[l][m, s].flatten(order = 'C'), \
                    cov = chsi[l][s])
                
        z2_z1s.append(z2_z1s_l)
    
    return z2_z1s


def draw_z_s_all_network(mu_s_c, sigma_s_c, mu_s_d, sigma_s_d, yc, eta_c, \
                         eta_d, S_1L, L, M):
    ''' Draw z^{(l)h} from both heads and then from the tail 
    mu_s_* (list of nd-arrays): The means of the Gaussians starting at each layer of the head *
    sigma_s_* (list of nd-arrays): The covariance matrices of the Gaussians of the head *
    yc (n x p_continuous): The continuous data
    eta_* (list of nb_layers elements of shape (K_l x r_{l-1}, 1)): mu parameters
                                                        for each layer of head *
    S_1L (dict): The number of paths starting at each layer
    L (dict):  The number of layers on each head or tail
    M (dict): The number of MC points to draw on each head or tail
    --------------------------------------------------------------------------
    returns (tuple of length 4): The latent variables existing in the network
    ''' 
    
    #============================
    # Continuous head. 
    #============================
    # The first z for the continuous head is actually the data, 
    # which we do not resimulate
    z_s_c, zc_s_c = draw_z_s(mu_s_c[1:], sigma_s_c[1:],\
                    eta_c[1:], M['c'][1:], False)

    yc_rep = np.repeat(yc[..., n_axis], S_1L['c'][0], -1)
    z_s_c = [yc_rep] + z_s_c 
    
    eta_rep = np.repeat(eta_c[0], S_1L['c'][1], axis = 0)
    yc_centered_rep = t(z_s_c[0] - eta_rep.T, (0, 2, 1))
    zc_s_c = [yc_centered_rep] + zc_s_c
      
    #============================
    # Discrete head. 
    #============================
    z_s_d, zc_s_d = draw_z_s(mu_s_d[:L['d']], sigma_s_d[:L['d']],\
                    eta_d[:L['d']], M['d'][:L['d']], True)
        
    #============================
    # Common tail 
    #============================
    # The samples of the tail are shared by both heads
    z_s_d = z_s_d + z_s_c[(L['c'] + 1):]
    zc_s_d = zc_s_d + zc_s_c[(L['c'] + 1):]
    
    return z_s_c, zc_s_c, z_s_d, zc_s_d   


def draw_z2_z1s_network(chsi_c, chsi_d, rho_c, rho_d, M, r_1L, L):
    ''' Draw z^{(l + 1)h} | z^{(l)h} from both heads and then from the tail 
    chsi_* (list of nd-arrays): The chsi parameters for all paths starting at each layer of head *
    rho_* (list of ndarrays): The rho parameters (covariance matrices) for
                                    all paths starting at each layer of head *
    M (dict of lists): The number of MC to draw on each layer
    r_1L (dict of lists): The dimension of each layer
    L (dict): The number of layers in the networks
    ---------------------------------------------------------------------
    returns (tuple of length 2): The latent variables of z^{(l + 1)h} | z^{(l)h} existing in the network
    ''' 
    
    # Draw z^{l+1} | z^l, s from head to tail for the continuous head
    z2_z1s_c = draw_z2_z1s(chsi_c, rho_c, M['c'], r_1L['c'])

    # Draw z^{l+1} | z^l, s from head to the first common layer for the discrete head
    z2_z1s_d = draw_z2_z1s(chsi_d[:L['d']], rho_d[:L['d']], \
                    M['d'][:(L['d'] + 1)], r_1L['d'][:(L['d'] + 1)])

    # Common z^{l+1} | z^l, s draws are shared by the two heads
    z2_z1s_d = z2_z1s_d + z2_z1s_c[(L['c'] + 1):]
    
    return z2_z1s_c, z2_z1s_d


#=============================================================================
# E Step functions
#=============================================================================

def fz_s(z_s, mu, sigma):
    ''' Compute p(z | s)
    z_s (list of ndarrays): The latent variables on each path
    mu (list of nd-arrays): The means of the Gaussians starting at each layer 
    sigma (list of nd-arrays): The covariance matrices of the Gaussians 
    ---------------------------------------------------------------------
    returns (list of ndarrays): Compute the density estimates of p(z^{(l)} | s^{(l)})
    '''

    epsilon = 1E-16
            
    L = len(z_s)
    M = [z_s[l].shape[0] for l in range(L)]
    S = [z_s[l].shape[-1] for l in range(L)]
    
    pz_s = []    
    for l in range(L):
        pz_sl = np.zeros((M[l], S[l]))  
        for s in range(S[l]):
            pz_sl[:, s] = mvnorm.pdf(z_s[l][:,:, s], \
                            mean = mu[l][s, :, 0], \
                            cov = sigma[l][s], allow_singular = True)

        # Normalization to check
        norm_cste = pz_sl.sum(0, keepdims = True) 
        norm_cste = np.where(norm_cste <= epsilon, epsilon, norm_cste) 

        pz_sl = pz_sl / norm_cste
        
        pz_sl = np.tile(pz_sl, (1, S[0]//S[l]))
        pz_s.append(pz_sl)
    
    return pz_s


def fz2_z1s(pzl1_ys, z2_z1s, chsi, rho, S):
    ''' Compute p(z^{(l)}| z^{(l-1)}, y) 
    pzl1_ys (ndarray): p(z1 |y, s)
    z2_z1s (list of ndarrays): z^{(l + 1)}| z^{(l)}, s
    chsi (list of nd-arrays): The chsi parameters for all paths starting at each layer
    rho (list of ndarrays): The rho parameters (covariance matrices) for
                                    all paths starting at each layer
    S (list of int): The number of paths starting at each layer
    -------------------------------------------------------------------------
    returns (list of ndarrays): p(z^{(l)}| z^{(l-1)}, y)
    '''
    
    epsilon = 1E-16
    L = len(z2_z1s)
    M = [z2_z1s[l].shape[0] for l in range(L)] + [z2_z1s[-1].shape[1]]
    
    if len(pzl1_ys) > 0: # For discrete head
        pz2_z1s = [pzl1_ys]
    else: # For continuous head
        pz2_z1s = []
            
    for l in range(L):
        pz2_z1sm = np.zeros((M[l], M[l + 1], S[l]))  
        for s in range(S[l]):
            for m in range(M[l]): 
                try:
                    pz2_z1sm[m, :, s] = mvnorm.pdf(z2_z1s[l][m, :, s], \
                                    mean = rho[l][m, s, :, 0], \
                                    cov = chsi[l][s], allow_singular = True)
                except:
                    print(chsi[l][s])
                    raise RuntimeError('Singular Matrix')
                    
        norm_cste = pz2_z1sm.sum(1, keepdims = True)
        norm_cste = np.where(norm_cste <= epsilon, epsilon, norm_cste) 

        pz2_z1sm = pz2_z1sm / norm_cste
        pz2_z1sm = np.tile(pz2_z1sm, (1, 1, S[0]//S[l]))
        pz2_z1s.append(pz2_z1sm)
        
    return pz2_z1s


def continuous_lik(yc, mu, sigma, w):
    ''' Compute p(y^C) and p(s^C | y^C)
    yc (n x p_continuous): The continuous data
    mu (list of nd-arrays): The means of the Gaussians starting at each layer 
    sigma (list of nd-arrays): The covariance matrices of the Gaussians 
    w (list): The flatten path probabilities     
    -------------------------------------------------------------------------
    returns (tuple of length 2): p(y^C) and p(s^C | y^C)
    '''
    epsilon = 1E-16
                
    #==========================
    # p(y|s), p(y,s)
    #==========================
    
    numobs = yc.shape[0]
    S0 = mu.shape[0]
    
    py_s = np.zeros((numobs, S0))
    for s in range(S0):
        try:
            py_s[:,s] = mvnorm.logpdf(yc, mu[s][:,0], sigma[s],\
                                      allow_singular = True)
        except:
            print(sigma[s])
            raise RuntimeError('Singular matrix')
        
    pys = np.log(w)[n_axis] + py_s  

    #==========================
    # Numeric stability block
    #==========================
    
    pys_max = np.max(pys, axis = 1, keepdims = True) 
    pys = pys - pys_max # For numeric stability
    pys = np.exp(pys) # Max-normalized p(y,s)
    
    #==========================
    # p(s|y), p(y)
    #==========================
    
    norm_cste = np.sum(pys, axis = 1, keepdims = True) 
    norm_cste = np.where(norm_cste <= epsilon, epsilon, norm_cste) 

    ps_y = pys / norm_cste  # Normalized p(s|y)
    py = np.exp(pys_max)[:, 0] * np.sum(pys, axis = 1) # p(y) = sum_{s} p(y,s)
    py_s = np.exp(py_s)
     
    # Numeric stability issues handling   
    ps_y = np.where(ps_y <= epsilon, epsilon, ps_y) 
    py = np.where(py == 0, epsilon, py)
                
    return ps_y, py_s, py[..., n_axis]

     
def fz_ys(pzl1_ys, pz2_z1s):
    ''' Compute p(z^{l} | y, s) in a recursive manner 
    pzl1_ys (ndarray): p(z1 |y, s)
    pz2_z1s (list of ndarrays): p(z^{(l)}| z^{(l-1)}, y)
    ------------------------------------------------------------
    returns (list of ndarrays): p(z^{l} | y, s)
    '''
    
    L = len(pz2_z1s) - 1
    
    if len(pzl1_ys) > 0: # For discrete head
        pz_ys = [pzl1_ys]
        
    else: # For continuous head
        pz_ys = [pz2_z1s[0]]
        
    for l in range(L):
        pz_ys_l = np.expand_dims(pz_ys[l], 2)
        pz2_z1s_l = pz2_z1s[l + 1][n_axis]
        
        # Was a mean before: error...
        pz_ys.append((pz_ys_l * pz2_z1s_l).sum(1))

    return pz_ys 


def fy_zs(pz_ys, py_s):
    ''' Compute p(y^h | z^{(l)h}, s^h)
    pz_ys (list of ndarrays): p(z^{(l)h} |y^h, s^h)
    py_s (ndarray): p(y^h| s^h)
    ----------------------------------------------------------------
    returns (list of ndarrays): p(y^h | z^{(l)h}, s^h)
    '''
    
    L = len(pz_ys)
    epsilon = 1E-16
    
    fy_zs = []
    for l in range(L):
        
        norm_cste = (pz_ys[l] * np.expand_dims(py_s, 1)).sum(0, keepdims = True)
        norm_cste = np.where(norm_cste <= epsilon, epsilon, norm_cste) 
        
        # Norm constant to check        
        fy_zsl = pz_ys[l] * np.expand_dims(py_s, 1) / norm_cste
        fy_zs.append(fy_zsl)
    
    return fy_zs


def fz_yCyDs(py_zs_c, pz_ys_d, py_s_c, M, S_1L, L):
    ''' Compute p(zt | yC, yD, sC, sD) for all common layers
    py_zs_c  (list of ndarrays): p(y^C | z^C,  s^C)
    pz_ys_d (list of ndarrays): p(z^D | y^D,  s^D)
    py_s_c (ndarray): p(y^C | s^C)
    M (dict): The number of MC points to draw on each head or tail
    S_1L (dict): The number of paths starting at each layer
    L (dict):  The number of layers on each head or tail
    ----------------------------------------------------------------
    returns (list of ndarrays): p(zt | yC, yD, sC, sD) 
    '''
    
    epsilon = 1E-16

    numobs = py_zs_c[0].shape[0]
    fz_yCyDs = []
    for l in range(L['t']):
        
        #====================================================
        # Reshaping
        #====================================================
            
        py_zs_c_l = py_zs_c[L['c'] + l].reshape(numobs, M['c'][L['c'] + l + 1], \
                                    S_1L['c'][0]// S_1L['t'][0], S_1L['t'][0],\
                                        order = 'C') 
        
        pz_ys_d_l = pz_ys_d[L['d'] + l].reshape(numobs, M['d'][L['d'] + l], \
                                    S_1L['d'][0]// S_1L['t'][0], S_1L['t'][0],\
                                        order = 'C')

        py_zs_c_l = np.expand_dims(py_zs_c_l, 3)            
        pz_ys_d_l = np.expand_dims(pz_ys_d_l, 2)
        
        # The shape is (numobs, M[l], S^C, S^D, S^t)
        fz_yCyDs_l = py_zs_c_l * pz_ys_d_l
        
        # To check for axis 4
        norm_cste = fz_yCyDs_l.sum((1, 3, 4), keepdims = True)
        norm_cste = np.where(norm_cste <= epsilon, epsilon, norm_cste) 
        
        fz_yCyDs_l = fz_yCyDs_l / norm_cste
        fz_yCyDs.append(fz_yCyDs_l)
    
    return fz_yCyDs
    
 
def fst_yCyD(py_zs_c, py_zs_d, pz_s_d, w_s_c, w_s_d, k_1L, L):
    ''' Compute p(s^L0 | y^C, y^D)
    py_zs_c (list of ndarrays): p(y^C | z^C, s^C)
    py_zs_d (list of ndarrays): p(y^D | z^D, s^D)
    pz_s_d (list of ndarrays): p(z^D | s^D)
    w_s_* (list of length s1): The path probabilities for all s in S^h 
                        of the given head or tail *
    k_1L (list of int): The number of component on each layer including the common layers
    L (dict):  The number of layers on each head or tail
    ----------------------------------------------------------------
    returns (list of ndarrays): p(s^t | y^C, y^D) 
    '''
    
    epsilon = 1E-16
    numobs = py_zs_c[0].shape[0]
            
    # 1st and 2nd term
    # p(y^h | zt^{(1)}, s^h, Theta) for h in {C, D}.
    py_zts_c = py_zs_c[L['c']] # shape = (numobs, Mt, Sc)
    Mt = py_zts_c.shape[1] # Nb of MC points of the first tail layer
    py_zts_c = py_zts_c.reshape(numobs, Mt, np.prod(k_1L['c'][:L['c'] + 1]),\
                                np.prod(k_1L['t']), order = 'C')
    
    py_zts_d = py_zs_d[L['d']] # shape = (numobs, Mt, Sd)
    # shape = (numobs, Mt, k['d'], S['t'])
    py_zts_d = py_zts_d.reshape(numobs, Mt, np.prod(k_1L['d'][:L['d']]),\
                                np.prod(k_1L['t']), order = 'C')
    
    # 3rd term
    # p(zt^{(1)} | s^t, Theta) 
    pz_s_t = pz_s_d[L['d']]    
    pz_s_t = pz_s_t.reshape(Mt, *k_1L['d'], order = 'C')
    # Quantities are tiled hence take the mean of the quantity not the sum
    idx_to_mean = tuple(range(1, L['d'] + 1)) 
    pz_s_t = pz_s_t.mean(idx_to_mean).reshape(Mt, -1, order = 'C')[n_axis]

    # 4th term
    # p(s^D | Theta) 
    ps_d = w_s_d.reshape(*k_1L['d'], order = 'C')
    idx_to_sum = tuple(range(L['d'], L['d'] + L['t']))
    ps_d = ps_d.sum(idx_to_sum).reshape(-1, order = 'C')[n_axis, n_axis, ..., n_axis] 
    
    
    # sum_s^C p(y^C | z^t, s^C, s^t) p(s^C, s^t)
    ps_ct = w_s_c.reshape(np.prod(k_1L['c'][:L['c'] + 1]),\
                                np.prod(k_1L['t']), order = 'C')
    pyst_zt_c = (py_zts_c * ps_ct[n_axis, n_axis]).sum(2) 
    
    # sum_s^D p(y^C | z^t, s^D, s^t) p(s^D)
    pysD_ztst_d = (py_zts_d * ps_d).sum(2)
    
    pst_yCyD = (pyst_zt_c * pysD_ztst_d * pz_s_t).sum(1)

    # Normalization 
    norm_cste = np.where(pst_yCyD == 0.0, epsilon, pst_yCyD) 
    pst_yCyD = pst_yCyD / norm_cste.sum(1, keepdims = True)

    return pst_yCyD
    

def E_step_DGMM_c(H, z_s, zc_s, z2_z1s, pz_ys, pz2_z1s, Sc, Lc):
    ''' Compute the expectations of the E step for all DGMM layers of the continuous head
    H (list of nb_layers elements of shape (K_l x r_l-1, r_l)): Lambda parameters
                                                                for each layer
    z_s (list of nd-arrays): z^{(l)} | s^l for all s^l and all l.
    zc_s (list of nd-arrays): (z^{(l)C} | s^{(l)C} - eta{k_l}^{(l)C} for all s^l and all l.
    z2_z1s (list of ndarrays): z^{(l + 1)}| z^{(l)}, s
    pz_ys (list of ndarrays): p(z^{l} | y, s)
    pz2_z1s (list of ndarrays): p(z^{(l)}| z^{(l-1)}, y)
    Sc (list of int): The number of paths starting at each layer of the continuous head
    Lc (list of int):  The number of layers on the continuous head
    ------------------------------------------------------------
    returns (tuple of ndarrays): E(z^{(l)} | y, s), E(z^{(l)}z^{(l+1)T} | y, s), 
            E(z^{(l+1)}z^{(l+1)T} | y, s), 
            E(e | y, s) with e = z^{(l)} - eta{k_l}^{(l)} - Lambda @ z^{(l + 1)}
    '''
    
    k = [H[l].shape[0] for l in range(Lc + 1)]
    
    Ez_ys = [t(z_s[0], (0, 2, 1)) ] # E(y | y ,s) = y
    E_z1z2T_ys = []
    E_z2z2T_ys = []
    EeeT_ys = []
    
    
    for l in range(Lc + 1):
        #print(l)
        # Broadcast the quantities to the right shape

        z1_s =  t(z_s[l], (0, 2, 1)) 
        z1_s = np.tile(z1_s, (1, Sc[0] // Sc[l], 1))[..., n_axis]  
        z1c_s = np.tile(zc_s[l], (1, np.prod(k[:l]), 1))

        z2_s = z_s[l + 1].transpose((0, 2, 1))
        z2_s = np.tile(z2_s, (1, np.prod(k[:l + 1]), 1))[..., n_axis]  
        
        pz1_ys = pz_ys[l - 1][..., n_axis]
        pz2_ys = pz_ys[l][..., n_axis] 
          
        H_formated = np.tile(H[l], (np.prod(k[:l]), 1, 1))
        H_formated = np.repeat(H_formated, Sc[l + 1], axis = 0)[n_axis] 
        
        #=========================================================
        # E(z^{l + 1} | z^{l}, s) = sum_M^{l + 1} z^{l + 1}  
        #=========================================================
        
        E_z2_z1s = z2_z1s[l].mean(1)
        E_z2_z1s = np.tile(E_z2_z1s, (1, Sc[0] // Sc[l], 1))
    
        if l == 0:
            Ez_ys_l = E_z2_z1s
        else:
            Ez_ys_l = (pz1_ys *  E_z2_z1s[n_axis]).sum(1)

        Ez_ys.append(Ez_ys_l)

        #=========================================================           
        # E(z^{l + 1}z^{l + 1}^T | z^{l}, s)   
        #=========================================================

        E_z2z2T_z1s = (z2_z1s[l][..., n_axis] @ \
                      np.expand_dims(z2_z1s[l], 3)).mean(1)  
        E_z2z2T_z1s = np.tile(E_z2z2T_z1s, (1, Sc[0] // Sc[l], 1, 1))

        #=========================================================        
        # E(z^{l + 1}z^{l + 1}^T | y, s)    
        #=========================================================

        if l == 0:
            E_z2z2T_ys.append(E_z2z2T_z1s)
        else:
            E_z2z2T_ys_l = (pz1_ys[..., n_axis] * \
                             E_z2z2T_z1s[n_axis]).sum(1)
            E_z2z2T_ys.append(E_z2z2T_ys_l)

        #=========================================================
        # E(z^{l}z^{l + 1}^T | y, s)                
        #=========================================================
          
        if l == 0: # E(y, z^{1} | y, s) = y @ E(z^{1} | y, s)
            # OK TO REMOVE THE SUM ?
            E_z1z2T_ys_l = z1_s @ np.expand_dims(Ez_ys_l, axis = 2)
        else:
            # To check
            E_z1z2T_ys_l = (pz1_ys[..., n_axis] * (z1_s[n_axis] @ \
                np.expand_dims(np.expand_dims(Ez_ys_l, axis = 1), 3))).sum(1)
        
        E_z1z2T_ys.append(E_z1z2T_ys_l)

        #=========================================================
        # E[((z^l - eta^l) - Lambda z^{l + 1})((z^l - eta^l) - Lambda z^{l + 1})^T | y, s]  
        #=========================================================
                             
        e = (np.expand_dims(z1c_s, 1) - t(H_formated @ z2_s, (3, 0, 1, 2)))[..., n_axis]

        eeT = e @ t(e, (0, 1, 2, 4, 3))
                
        if l == 0:
            EeeT_ys_l = (pz2_ys[...,n_axis] * eeT).sum(1)
        else:   
            pz1z2_ys = np.expand_dims(pz_ys[l - 1], 2) * pz2_z1s[l][n_axis]
            pz1z2_ys = pz1z2_ys[..., n_axis, n_axis]
            EeeT_ys_l = (pz1z2_ys * eeT[n_axis]).sum((1, 2))
        
        EeeT_ys.append(EeeT_ys_l)
        
    
    return Ez_ys, E_z1z2T_ys, E_z2z2T_ys, EeeT_ys


def E_step_DGMM_d(zl1_ys, H, z_s, zc_s, z2_z1s, pz_ys, pz2_z1s, Sd, Ld):
    ''' Compute the expectations of the E step for all DGMM layers of the discrete head
    zl1_ys ((M1, numobs, r1, S1) nd-array): z^{(1)} | y, s
    H (list of nb_layers elements of shape (K_l x r_l-1, r_l)): Lambda parameters
                                                                for each layer
    z_s (list of nd-arrays): zl | s^l for all s^l and all l.
    zc_s (list of nd-arrays): (zl | s^l) - eta{k_l}^{(l)} for all s^l and all l.
    z2_z1s (list of ndarrays): z^{(l + 1)}| z^{(l)}, s
    pz_ys (list of ndarrays): p(z^{l} | y, s)
    pz2_z1s (list of ndarrays): p(z^{(l)}| z^{(l-1)}, y)
    Sd (list of int): The number of paths starting at each layer of the discrete head
    Ld (list of int):  The number of layers on the discrete head   
    ------------------------------------------------------------
    returns (tuple of ndarrays): E(z^{(l)} | y, s), E(z^{(l)}z^{(l+1)T} | y, s), 
            E(z^{(l+1)}z^{(l+1)T} | y, s), 
            E(e | y, s) with e = z^{(l)} - eta{k_l}^{(l)} - Lambda @ z^{(l + 1)}
    '''
    
    k = [H[l].shape[0] for l in range(Ld)]
    
    Ez_ys = []
    E_z1z2T_ys = []
    E_z2z2T_ys = []
    EeeT_ys = []
    
    if len(zl1_ys) != 0:
        Ez_ys.append(t(np.mean(zl1_ys, axis = 0), (0, 2, 1))) 
    
    for l in range(Ld):
        
        #=========================================================
        # Broadcast the quantities to the right shape
        #=========================================================

        z1_s = z_s[l].transpose((0, 2, 1))[..., n_axis]  
        z1_s = np.tile(z1_s, (1, np.prod(k[:l]), 1, 1)) # To recheck when L > 3

        z1c_s = np.tile(zc_s[l], (1, np.prod(k[:l]), 1))
        
        z2_s =  t(z_s[l + 1], (0, 2, 1)) 
        z2_s = np.tile(z2_s, (1, Sd[0] // Sd[l + 1], 1))[..., n_axis]  
        
        pz1_ys = pz_ys[l][..., n_axis] 
        
        H_formated = np.tile(H[l], (np.prod(k[:l]), 1, 1))
        H_formated = np.repeat(H_formated, Sd[l + 1], axis = 0)[n_axis] 
 
        #=========================================================
        # E(z^{l + 1} | z^{l}, s) = sum_M^{l + 1} z^{l + 1}  
        #=========================================================

        E_z2_z1s = z2_z1s[l].mean(1)
        E_z2_z1s = np.tile(E_z2_z1s, (1, Sd[0] // Sd[l], 1))
    
        #=========================================================           
        # E(z^{l + 1}z^{l + 1}^T | z^{l}, s)   
        #=========================================================

        E_z2z2T_z1s = (z2_z1s[l][..., n_axis] @ \
                      np.expand_dims(z2_z1s[l], 3)).mean(1)  
        E_z2z2T_z1s = np.tile(E_z2z2T_z1s, (1, Sd[0] // Sd[l], 1, 1))

        #==========================================================
        #### E(z^{l + 1} | y, s) = integral_z^l [ p(z^l | y, s) * E(z^{l + 1} | z^l, s) ] 
        #==========================================================
        
        E_z2_ys_l = (pz1_ys * E_z2_z1s[n_axis]).sum(1)   
        Ez_ys.append(E_z2_ys_l)

        #=========================================================
        # E(z^{l}z^{l + 1}^T | y, s)                
        #=========================================================
        
        E_z1z2T_ys_l = (pz1_ys[..., n_axis] * \
                           (z1_s @ np.expand_dims(E_z2_z1s, 2))[n_axis]).sum(1)
        E_z1z2T_ys.append(E_z1z2T_ys_l)
                            
        #=========================================================        
        # E(z^{l + 1}z^{l + 1}^T | y, s)    
        #=========================================================

        E_z2z2T_ys_l = (pz1_ys[..., n_axis] * E_z2z2T_z1s[n_axis]).sum(1)
        E_z2z2T_ys.append(E_z2z2T_ys_l)
    
        #=========================================================
        # E[((z^l - eta^l) - Lambda z^{l + 1})((z^l - eta^l) - Lambda z^{l + 1})^T | y, s]  
        #=========================================================
                 
        pz1z2_ys = np.expand_dims(pz_ys[l], 2) * pz2_z1s[l + 1][n_axis]
        pz1z2_ys = pz1z2_ys[..., n_axis, n_axis]
                
        e = (np.expand_dims(z1c_s, 1) - t(H_formated @ z2_s, (3, 0, 1, 2)))[..., n_axis]
        eeT = e @ t(e, (0, 1, 2, 4, 3))
                
        EeeT_ys_l = (pz1z2_ys * eeT[n_axis]).sum((1, 2))
                
        EeeT_ys.append(EeeT_ys_l)
    
    return Ez_ys, E_z1z2T_ys, E_z2z2T_ys, EeeT_ys


def E_step_DGMM_t(H, z_s, zc_s, z2_z1s, pz_ys, pz2_z1st, S_1L, L, k_1L):
    ''' Compute the expectations of the E step for all DGMM layers of the commont tail
    zl1_ys ((M1, numobs, r1, S1) nd-array): z^{(1)} | y, s
    H (list of nb_layers elements of shape (K_l x r_l-1, r_l)): Lambda parameters
                                                                for each layer
    z_s (list of nd-arrays): zl | s^l for all s^l and all l.
    zc_s (list of nd-arrays): (zl | s^l) - eta{k_l}^{(l)} for all s^l and all l.
    z2_z1s (list of ndarrays): z^{(l + 1)}| z^{(l)}, s
    pz_ys (list of ndarrays): p(z^{l} | y, s)
    pz2_z1st (list of ndarrays): p(z^{(l)}| z^{(l-1)}, y)
    S_1L (dict): The number of paths starting at each layer in the network
    L (dict):  The number of layers on each head or tail
    k_1L (list of int): The number of component on each layer including the common layers
    ------------------------------------------------------------
    returns (tuple of ndarrays): E(z^{(l)} | y, s), E(z^{(l)}z^{(l+1)T} | y, s), 
            E(z^{(l+1)}z^{(l+1)T} | y, s), 
            E(e | y, s) with e = z^{(l)} - eta{k_l}^{(l)} - Lambda @ z^{(l + 1)}
    '''
    
    Ez_ys = []
    E_z1z2T_ys = []
    E_z2z2T_ys = []
    EeeT_ys = []
    
    kt = k_1L['t']
    
    for l in range(L['t'] - 1):
        
        #===============================================
        # Broadcast the quantities to the right shape
        #===============================================
                
        z1_s = z_s[l].transpose((0, 2, 1))
        z1_s = np.expand_dims(np.expand_dims(z1_s, 1), 2)
        z1_s = np.tile(z1_s, (1, 1, 1, S_1L['t'][0] // S_1L['t'][l], 1)) 
    
        z1c_s = np.expand_dims(np.expand_dims(zc_s[l], 1), 2)
        z1c_s = np.tile(z1c_s, (1, 1, 1, S_1L['t'][0] // S_1L['t'][l], 1)) 
        
        z2_s =  t(z_s[l + 1], (0, 2, 1)) 
        z2_s = np.expand_dims(np.expand_dims(z2_s, 1), 2)
        z2_s = np.tile(z2_s, (1, 1, 1, S_1L['t'][0] // S_1L['t'][l + 1], 1))

        pz1_ys = pz_ys[l][..., n_axis] 
        pz2_ys = pz_ys[l + 1][..., n_axis] 
        pz2_z1s = np.expand_dims(np.expand_dims(pz2_z1st[l], 2), 2)[n_axis]
        
        if l == 0:
            Ez_ys_l = (pz1_ys * z1_s[n_axis]).sum(1)
            Ez_ys.append(Ez_ys_l)         

        H_formated = np.tile(H[l], (np.prod(kt[: l]), 1, 1))
        H_formated = np.repeat(H_formated, S_1L['t'][l + 1], axis = 0)
        H_formated = H_formated[n_axis, n_axis, n_axis] 


        #===============================================    
        # Compute the expectations
        #===============================================

        #=========================================================
        # E(z^{l + 1} | z^{l}, s) = sum_M^{l + 1} z^{l + 1}  
        #=========================================================
        
        E_z2_z1s = z2_z1s[l].mean(1)
        E_z2_z1s = np.tile(E_z2_z1s, (1, S_1L['t'][0] // S_1L['t'][l], 1))

        #=========================================================           
        # E(z^{l + 1}z^{l + 1}^T | z^{l}, s)   
        #=========================================================
    
        E_z2z2T_z1s = (z2_z1s[l][..., n_axis] @ \
                      np.expand_dims(z2_z1s[l], 3)).mean(1)  
        E_z2z2T_z1s = np.tile(E_z2z2T_z1s, (1, S_1L['t'][0] // S_1L['t'][l], 1, 1))
        
        # Create a new axis for the information coming from the discrete head
        E_z2_z1s = np.expand_dims(np.expand_dims(E_z2_z1s, 1), 1)
        E_z2z2T_z1s = np.expand_dims(np.expand_dims(E_z2z2T_z1s, 1), 1)
               
        #==========================================================
        # E(z^{l + 1} | y, s) = integral_z^l [ p(z^l | y, s) * E(z^{l + 1} | z^l, s) ] 
        #==========================================================
        
        E_z2_ys_l = (pz2_ys * z2_s[n_axis]).sum(1) 
        Ez_ys.append(E_z2_ys_l)
 
        #=========================================================
        # E(z^{l}z^{l + 1}^T | y, s)                
        #=========================================================
 
        E_z1z2T_ys_l = (pz1_ys[..., n_axis] * \
                           (z1_s[..., n_axis] @ np.expand_dims(E_z2_z1s, 4))[n_axis]).sum(1)
        E_z1z2T_ys.append(E_z1z2T_ys_l)

        #=========================================================        
        # E(z^{l + 1}z^{l + 1}^T | y, s)    
        #=========================================================
                            
        E_z2z2T_ys_l = (pz1_ys[..., n_axis] * E_z2z2T_z1s[n_axis]).sum(1)
        E_z2z2T_ys.append(E_z2z2T_ys_l)

        #=========================================================
        # E[((z^l - eta^l) - Lambda z^{l + 1})((z^l - eta^l) - Lambda z^{l + 1})^T | y, s]  
        #=========================================================
                 
        pz1z2_ys = np.expand_dims(pz_ys[l], 2) * pz2_z1s
        pz1z2_ys = pz1z2_ys[..., n_axis, n_axis]
                
        e = np.expand_dims(z1c_s, 1) - t(H_formated @ z2_s[..., n_axis], (5, 0, 1, 2, 3, 4))
        e = e[..., n_axis]
        eeT = e @ t(e, (0, 1, 2, 3, 4, 6, 5))
                
        eeT = eeT[n_axis]
        EeeT_ys_l = (pz1z2_ys * eeT).sum((1, 2))
                
        EeeT_ys.append(EeeT_ys_l)
    
    return Ez_ys, E_z1z2T_ys, E_z2z2T_ys, EeeT_ys


def M_step_DGMM(Ez_ys, E_z1z2T_ys, E_z2z2T_ys, EeeT_ys, ps_y, H_old, k, L_1Lh, rh):
    ''' 
    Compute the estimators of eta, Lambda and Psi for all components and all layers of both heads
    Ez_ys (list of ndarrays): E(z^{(l)} | y, s) for all (l,s)
    E_z1z2T_ys (list of ndarrays):  E(z^{(l)}z^{(l+1)T} | y, s) 
    E_z2z2T_ys (list of ndarrays):  E(z^{(l+1)}z^{(l+1)T} | y, s) 
    EeeT_ys (list of ndarrays): E(z^{(l+1)}z^{(l+1)T} | y, s), 
            E(e | y, s) with e = z^{(l)} - eta{k_l}^{(l)} - Lambda @ z^{(l + 1)}
    ps_y ((numobs, S) nd-array): p(s | y) for all s in Omega
    H_old (list of ndarrays): The previous iteration values of Lambda estimators
    k (dict): The number of component on each layer
    L_1Lh (list of int): The number layers starting from the head h until the end of the common tail
    rh (list): The dimension of each layer of the head h
    --------------------------------------------------------------------------
    returns (list of ndarrays): The new estimators of eta, Lambda and Psi 
                                            for all components and all layers
    '''
    epsilon = 1E-16
    
    Lh = len(E_z1z2T_ys)
    numobs = len(Ez_ys[0])

    eta = []
    H = []
    psi = []
    
    for l in range(Lh):

        #===============================================
        # Broadcast the quantities to the right shape
        #===============================================
        
        Ez1_ys_l = Ez_ys[l].reshape(numobs, *k, rh[l], order = 'C')
        Ez2_ys_l = Ez_ys[l + 1].reshape(numobs, *k, rh[l + 1], order = 'C')
        E_z1z2T_ys_l = E_z1z2T_ys[l].reshape(numobs, *k, rh[l], rh[l + 1], order = 'C')
        E_z2z2T_ys_l = E_z2z2T_ys[l].reshape(numobs, *k, rh[l + 1], rh[l + 1], order = 'C')
        EeeT_ys_l = EeeT_ys[l].reshape(numobs, *k, rh[l], rh[l], order = 'C')
        
        # Sum all the path going through the layer
        idx_to_sum = tuple(set(range(1, L_1Lh)) - set([l + 1]))
        ps_yl = ps_y.reshape(numobs, *k, order = 'C').sum(idx_to_sum)[..., n_axis, n_axis]  

        # Compute common denominator    
        den = ps_yl.sum(0)
        den = np.where(den < epsilon, epsilon, den)  
        
        #===============================================
        # eta estimator
        #===============================================

        eta_num = Ez1_ys_l.sum(idx_to_sum)[..., n_axis] -\
            H_old[l][n_axis] @ Ez2_ys_l.sum(idx_to_sum)[..., n_axis]
        eta_new = (ps_yl * eta_num).sum(0) / den
        
        eta.append(eta_new)

        #===============================================
        # Lambda estimator
        #===============================================

        H_num = E_z1z2T_ys_l.sum(idx_to_sum) - \
            eta_new[n_axis] @ np.expand_dims(Ez2_ys_l.sum(idx_to_sum), 2)
        
        H_new = (ps_yl * H_num  @ pinv(E_z2z2T_ys_l.sum(idx_to_sum))).sum(0) / den 
        H.append(H_new)

        #===============================================
        # Psi estimator
        #===============================================

        psi_new = (ps_yl * EeeT_ys_l.sum(idx_to_sum)).sum(0) / den
        psi.append(psi_new)

    return eta, H, psi

# L and L_1L are redundant
def M_step_DGMM_t(Ez_ys, E_z1z2T_ys, E_z2z2T_ys, EeeT_ys, \
                  ps_y_c, ps_y_d, pst_yCyD, H_old, S_1L, k_1L, L_1L, L, rh):
    ''' 
    Compute the estimators of eta, Lambda and Psi for all components and all layers of the tail
    Ez_ys (list of ndarrays): E(z^{(l)} | y, s) for all (l,s)
    E_z1z2T_ys (list of ndarrays):  E(z^{(l)}z^{(l+1)T} | y, s) 
    E_z2z2T_ys (list of ndarrays):  E(z^{(l+1)}z^{(l+1)T} | y, s) 
    EeeT_ys (list of ndarrays): E(z^{(l+1)}z^{(l+1)T} | y, s), 
            E(e | y, s) with e = z^{(l)} - eta{k_l}^{(l)} - Lambda @ z^{(l + 1)}
    ps_y_* ((numobs, S) nd-array): p(s | y) for all s in Omega^*
    pst_yCyD ((numobs, S) nd-array): p(s^t | y) for all s in the tail
    H_old (list of nb_layers elements of shape (K_l x r_l-1, r_l)): Lambda 
                        estimators of the previous iteration for each layer
    S_1L (dict): The number of paths starting at each layer until the common tail
    k_1L (list of int): The number of component on each layer including the common tail
    L_1L (dict): The number of layers where the lists include the heads and the tail layers
    L (dict): The number of layers in the networks (to delete in future versions)
    rh (list): The dimension of each layer of the head h
    --------------------------------------------------------------------------
    returns (tuple of length 4): The new estimators of eta, Lambda and Psi 
                                            for all components and all layers
                                            and the associated latent expectancy 
    '''
    
    Lh = len(E_z1z2T_ys)
    numobs = len(Ez_ys[0])
    
    eta = []
    H = []
    psi = []
    
    #==========================================================================
    # Broadcast path probabilities
    #==========================================================================

    psc_y = ps_y_c.reshape(numobs, S_1L['c'][0] // S_1L['t'][0],\
                 S_1L['t'][0], order = 'C')
    psd_y = ps_y_d.reshape(numobs, S_1L['d'][0] // S_1L['t'][0],\
                 S_1L['t'][0], order = 'C')
    
    psc_y = psc_y.sum(-1, keepdims = True)
    psc_y = np.expand_dims(psc_y, 2)
    
    psd_y = psd_y.sum(-1, keepdims = True)
    psd_y = np.expand_dims(psd_y, 1)
    
    # p(sC, sD, st) = p(sC | yC) p(sD | yD) p(st | yC, yD)
    # Add a normalization ?
    psCsDst_y = psc_y * psd_y * np.expand_dims(np.expand_dims(pst_yCyD, 1), 1)
    
    Ezst_y = []

    for l in range(Lh):
        #======================================================================
        # Compute the full expectations multiplying by p(sC, sD, st | yC, yD)
        #======================================================================

        Ez1_yst = (psCsDst_y[..., n_axis] * Ez_ys[l]).sum((1, 2))
        Ezst_y.append(deepcopy(Ez1_yst))
        
        Ez2_yst = (psCsDst_y[..., n_axis] * Ez_ys[l + 1]).sum((1, 2))
        
        E_z1z2T_yst = (psCsDst_y[..., n_axis, n_axis] * \
                                       E_z1z2T_ys[l]).sum((1, 2))
        
        E_z2z2T_yst = (psCsDst_y[..., n_axis, n_axis] * \
                                       E_z2z2T_ys[l]).sum((1, 2)) 
            
        EeeT_yst = (psCsDst_y[..., n_axis, n_axis] * \
                                       EeeT_ys[l]).sum((1, 2)) 
            
        #======================================================================
        # Broadcast the expectations
        #======================================================================            
        
        Ez1_yst = Ez1_yst.reshape(numobs, *k_1L['t'], rh[l], order = 'C')
        Ez2_yst = Ez2_yst.reshape(numobs, *k_1L['t'], rh[l + 1], order = 'C')
        E_z1z2T_yst = E_z1z2T_yst.reshape(numobs, *k_1L['t'], rh[l], rh[l + 1], order = 'C')
        E_z2z2T_yst = E_z2z2T_yst.reshape(numobs, *k_1L['t'], rh[l + 1], rh[l + 1], order = 'C')
        EeeT_yst = EeeT_yst.reshape(numobs, *k_1L['t'], rh[l], rh[l], order = 'C')
   
        # Sum all the paths going through the components of the layer
        idx_to_sum = tuple(set(range(1, L_1L['t'] + 1)) - set([l + 1]))

        #======================================================================
        # Compute the estimators
        #======================================================================            

        # One common denominator for all estimators
        den = psCsDst_y.sum((1, 2))
        den = den.reshape(numobs, *k_1L['t'], order = 'C').sum(idx_to_sum) 
        den = den.sum(0)[..., n_axis, n_axis]
        den = np.where(den < 1E-14, 1E-14, den)  
        
        # eta estimator
        eta_num = Ez1_yst.sum(idx_to_sum)[..., n_axis] -\
            H_old[l][n_axis] @ Ez2_yst.sum(idx_to_sum)[..., n_axis]
        eta_new =  eta_num.sum(0) / den
        
        eta.append(eta_new)
    
        # Lambda estimator
        H_num = E_z1z2T_yst.sum(idx_to_sum) - \
            eta_new[n_axis] @ np.expand_dims(Ez2_yst.sum(idx_to_sum), 2)
        
        try:
            H_new = (H_num @ pinv(E_z2z2T_yst.sum(idx_to_sum), rcond=1e-2)).sum(0) / den 
        except:
            print(E_z2z2T_yst.sum(idx_to_sum))
            raise RuntimeError('Overflow ?')
        H.append(H_new)

        # Psi estimator
        psi_new = EeeT_yst.sum(idx_to_sum).sum(0) / den
        psi.append(psi_new)

    return eta, H, psi, Ezst_y