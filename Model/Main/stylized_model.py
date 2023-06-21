# -*- coding: utf-8 -*-
"""
Created on Wed May 25 11:02:16 2022

@author: fcv495
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 15:04:05 2021

@author: fcv495
"""

#%autoreload


import os
os.environ["NUMBA_PARFOR_MAX_TUPLE_SIZE"] = "200"
import sys
sys.path.append("..")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,'..')

import time
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt   

import utils
import figs

import numba as nb
nb.set_num_threads(4)

%load_ext autoreload
%autoreload 2


#%% params

T = 300 
beta = 0.98
alpha=0.5
phi = 1.5 
kappa_w = 0.1
mu = 1.1
inv_frisch = 0.5 

#%%

G_r_Q = -np.eye(T)
G_r_Q += np.diag(np.ones(T-1), 1) 
G_r_Q *= - 1
gamma_mat = np.eye(T)
gamma_mat -= np.diag(np.ones(T-1), 1) * beta

xi = inv(1/(1-alpha) * ((phi-1)*kappa_w/mu*alpha*np.eye(T) + gamma_mat @ G_r_Q ))

#%%
# UIP

dQ = 1 * 0.8**(np.arange(T))

dr_act = np.zeros(T)
for k in range(T):
    t =  (T-1)-k
    if t == T-1:
        dr_act[t] = 0
    else:
        dr_act[t] = dQ[t+1] - dQ[t]
        
dr_test = - G_r_Q @ dQ #   G(r,Q) = - B_mat

plt.plot(dr_act[:40])
plt.plot(dr_test[:40],'--')
plt.show()


G_Q_Y = -xi * ((phi-1) * kappa_w/inv_frisch)

g_Q_Y = xi * ((phi-1) * kappa_w/inv_frisch)





#%%

phi = 1.5
xi = inv(1/(1-alpha) * ((phi-1)*kappa_w/mu*alpha*np.eye(T) - beta_mat @ B_mat ))



xi_peg =  inv(1/(1-alpha) * (kappa_w/mu*alpha*np.eye(T) + beta_mat @ B_mat ))


#%%
xi = inv(1/(1-alpha) * ((phi-1)*kappa_w/mu*alpha*np.eye(T) - beta_mat @ B_mat ))


#%%

# unknowns: Q, piH, w

dpiH = kappa_w *(dY/inv_frisch - mu*dw) + beta*dpiH_p # NKPC 
dr = dQ_p - dQ # real UIP 
dw = (dPh-dP)/mu
dr = phi*dpiH_p - dpi_p # Taylor rule 
dpiH_p = - 1/(1-alpha) * (dQ_p - dQ)







