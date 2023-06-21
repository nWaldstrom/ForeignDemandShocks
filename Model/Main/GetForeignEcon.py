import numpy as np
import utils
from ForeignEcon.ForeignEcon import NKModelClass
from copy import deepcopy

# solve foreign economy  
def get_foreign_econ(shock='betaF', do_print=False, shocksize=-0.001, persistence=None, upars={}):
    model_foreign = NKModelClass(name='baseline',par=upars)
    for k in upars:
        setattr(model_foreign.par,k,upars[k])
    model_foreign.find_ss()
    
    # update shock persistence 
    if persistence is None:
        pass
    else:
        for var in model_foreign.shocks:
            setattr(model_foreign.par,'rho_'+var, persistence)
    
    model_foreign.compute_jacs(do_print=do_print,skip_shocks=True,skip_hh=True)
    utils.Create_shock(shock, model_foreign, shocksize, absval=True)
    model_foreign.transition_path(do_print=do_print)
    return model_foreign


def create_foreign_shock(model, model_foreign):   

    # check that paths have same length in across models 
    assert model.par.T == model_foreign.par.T 

    for i in model.shocks:
        arraypath = getattr(model.path, i)
        ssval = getattr(model.ss,i)
        arraypath[:,:] = ssval
            
    for i in ['C_s', 'piF_s', 'iF_s']:      
        T_half = model.par.T//2
        dX = getattr(model_foreign.path,i)[0,:] 
        arraypath = getattr(model.path, i)
        arraypath[:,:] = dX[np.newaxis,:]
        arraypath[:,T_half:] = dX[T_half] # truncate shock after T/2 periods for faster convergence 

    # beta shock 
    T_half = model.par.T//2
    dX = model_foreign.path.betaF[0,:]
    arraypath = getattr(model.path, 'eps_beta')
    arraypath[:,:] = model.ss.eps_beta + model.par.beta_corr *  (dX[np.newaxis,:] - model_foreign.ss.betaF) # 1 in steady state 
    arraypath[:,T_half:] = 1.0


    