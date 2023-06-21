import numpy as np
from numba import njit
from scipy.stats import norm

# create grids

from consav.grids import equilogspace
# from consav.markov import log_rouwenhorst
from utils import markov_rouwenhorst



def asset_grid(amin, amax, n):
    # find maximum ubar of uniform grid corresponding to desired maximum amax of asset grid
    ubar = np.log(1 + np.log(1 + amax - amin))
    
    # make uniform grid
    u_grid = np.linspace(0, ubar, n)
    
    # double-exponentiate uniform grid and add amin to get grid from amin to amax
    return amin + np.exp(np.exp(u_grid) - 1) - 1

def create_grids_full(model):
    """ create grids """

    # note: only fills out already allocated arrays
    par = model.par
    ss = model.ss
    
    # a. beta
    par.beta_grid[:] = np.linspace(par.beta_mean-par.beta_delta,par.beta_mean+par.beta_delta,par.Nbeta)
        
    # b. a
    par.a_grid[:] = asset_grid(par.a_min,par.a_max,par.Na)
    if par.a_min < 0:
         par.a_grid[:] = a_grid_borrow_prem(par.a_grid)    
    if par.a_min < 0:
        par.a_grid[:] = a_grid_new(ss,par)
    # i. earnings 
    # create e grids
    par.e_grid_ss[:],par.e_ergodic_ss[:],par.e_trans_ss[:,:] = markov_rouwenhorst(par.rho_e,par.sigma_e,par.Ne)
     
    par.s_set[:par.Ne] = 0 
    par.s_set[par.Ne:] = 1 
    
    par.sT_vec = np.array([0.5+par.sTransferT,1.5-par.sTransferT]) 
    
    # ii. sectors 
    par.s_ergodic_ss[0] = par.sT
    par.s_ergodic_ss[1] = 1-par.sT

    Markov = np.identity(par.Ns)

    PTT = 1.0
    Markov[0,0] = PTT
    Markov[0,1] = 1-PTT
    Markov[1,1] = 1- (1-PTT) * par.sT/(1-par.sT)
    Markov[1,0] = 1-Markov[1,1]
    
    par.s_trans_ss = Markov 

    par.T_NT_trans[0,0] = model.par.sT
    par.T_NT_trans[1,1] = 1-model.par.sT        

    # d. Construct marov matricies 
    for i_s in range(par.Ns):
        par.z_ergodic_ss[i_s*par.Ne:(1+i_s)*par.Ne] = par.e_ergodic_ss * par.s_ergodic_ss[i_s]
        ss.z_trans[:,:,:] = np.kron(par.s_trans_ss, par.e_trans_ss)[np.newaxis,:,:]  


    create_z_grid_ss(par, ss.wnT, ss.wnNT, par.z_grid_ss)
    create_LT_grid(par.LT_grid, par.z_grid_ss, par.z_ergodic_ss, par)
    

        
def beta_grid_broadc(par):
   for i_s in range(par.Ns):
       par.beta_grid_s[i_s*par.Ne:(1+i_s)*par.Ne] = par.beta_s[i_s]


def create_grids_path_full(model):
        """ create grids for solving backwards along the transition path """

        # note: can use the full path of all inputs to the household block

        par = model.par
        path = model.path

        for t in range(par.T):
            par.z_grid_path[t,:] = par.z_grid_ss
            par.z_trans_path[t,:,:] = par.z_trans_ss
  

#@njit 
def create_LT_grid(LT_grid, z_grid, z_ergodic, par):
    # hardwired incidence rules are proportional to skill; scale does not matter 
    for i_z in range(par.Nz):
        LT_grid[i_z] = 1 / np.sum(z_ergodic * z_grid) * z_grid[i_z]

#@njit 
def create_z_grid_ss_sep_inc(par, wnT, wnNT, z):
    for i_s in range(par.Ns):
        if i_s == 0:
            z[i_s*par.Ne:(1+i_s)*par.Ne] = par.e_grid_ss   
        elif i_s == 1:
            z[i_s*par.Ne:(1+i_s)*par.Ne] = par.e_grid_ss  

#@njit 
def create_z_grid_ss_pooled_inc(par, wnT, wnNT, z):
    I = wnT * par.sT + wnNT * (1-par.sT)
    for i_s in range(par.Ns):
        if i_s == 0:
            z[i_s*par.Ne:(1+i_s)*par.Ne] = par.e_grid_ss * I
        elif i_s == 1:
            z[i_s*par.Ne:(1+i_s)*par.Ne] = par.e_grid_ss * I
        
#@njit        
def create_z_grid_path_sep_inc(par, wnT, wnNT, z):
    for i_s in range(par.Ns):
        if i_s == 0:
            z[i_s*par.Ne:(1+i_s)*par.Ne] = par.e_grid_ss * wnT 
        elif i_s == 1:
            z[i_s*par.Ne:(1+i_s)*par.Ne] = par.e_grid_ss * wnNT

#@njit        
def create_z_grid_path_pooled_inc(par, wnT, wnNT, z):
    for t in range(par.T):  
        I = wnT[t] * par.sT + wnNT[t] * (1-par.sT)
        for i_s in range(par.Ns):
            if i_s == 0:
                z[t,i_s*par.Ne:(1+i_s)*par.Ne] = par.e_grid_ss * I
            elif i_s == 1:
                z[t,i_s*par.Ne:(1+i_s)*par.Ne] = par.e_grid_ss * I
                                
                
#@njit                     
def create_z_grid_ss(par, wnT, wnNT, z):         
    if par.HA_PooledInc:
        create_z_grid_ss_pooled_inc(par, wnT, wnNT, z)
    else:
        create_z_grid_ss_sep_inc(par, wnT, wnNT, z)
        
#@njit                     
def create_z_grid_path(par, wnT, wnNT, z):         
    if par.HA_PooledInc:
        create_z_grid_path_pooled_inc(par, wnT, wnNT, z)
    else:
        create_z_grid_path_sep_inc(par, wnT, wnNT, z)
        
        
                
def a_grid_borrow_prem(grid):
    amin = grid.flat[np.abs(grid - 0).argmin()]
    if amin>0:
        aneg = np.abs(grid - 0).argmin() -1
        apos = np.abs(grid - 0).argmin()
    else:
        aneg = np.abs(grid - 0).argmin()
        apos = np.abs(grid - 0).argmin() + 1  
    
    
    grid[aneg] = -1e-08
    grid[apos] = 1e-08
    return grid           

def nonlinspace(amax, n, phi, amin=0):
    """Create grid between amin and amax. phi=1 is equidistant, phi>1 dense near amin. Extra flexibility may be useful in non-convex problems in which policy functions have nonlinear (even non-monotonic) sections far from the borrowing limit."""
    a_grid = np.zeros(n)
    a_grid[0] = amin
    for i in range(1, n):
        a_grid[i] = a_grid[i-1] + (amax - a_grid[i-1]) / (n-i)**phi 
    return a_grid
        
def a_grid_new(ss,par):     
    Na = par.Na
    amax = par.a_max
    amin = par.a_min
    
    a_grid_test = np.zeros(Na)

    
    N_lin = round(0.25*par.Na)
    N_lin_break = 0
    a_grid_test[:-N_lin] = np.linspace(amin,N_lin_break,Na-N_lin)
    a_grid_test[Na-N_lin:] = nonlinspace(amax, N_lin, 1.9, amin=N_lin_break)
    a_grid_test = a_grid_borrow_prem(a_grid_test)    
    return a_grid_test
                
            
        