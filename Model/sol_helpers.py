import numpy as np
import numba as nb
from copy import deepcopy
from IHANKModel import HANKModelClass
import GetForeignEcon
import utils

def sol_models(upars={}, upars_HH=None, scalevar='C_s', cumeffect=False, size=0.01, do_reduction=False, scale_kwargs=None, HHs = ['HA','RA-IM'], model_foreign=None, shock='foreign'):
    models = {}
    compiled = False
    for i,HH in enumerate(HHs):
        models[HH] = {}
        if compiled:
            pass
        else:
            model_ = HANKModelClass(name=HH)
        print(HH)
        model_.par.HH_type = HH
        for param in upars:
            setattr(model_.par,param,upars[param])
        if upars_HH is not None:
            for param in upars_HH[HH]:
                setattr(model_.par,param,upars_HH[HH][param])

        model_.find_ss()
        if model_.par.HH_type == 'HA':
            model_.compute_jacs(do_print=False,skip_shocks=True)
        else:
            model_.compute_jacs(do_print=False,skip_shocks=True,skip_hh=True)
        if shock == 'foreign':
            GetForeignEcon.create_foreign_shock(model_, model_foreign)
        else:
            utils.Create_shock(shock, model_, 0.001, absval=True)
        model_.transition_path(do_print=False)
        if scale_kwargs is None:
            utils.scaleshock(scalevar, model_, size=size, cumeffect=cumeffect)
        else:
            utils.scaleshock(model=model_, **scale_kwargs)            
        local_model = deepcopy(model_)
        if do_reduction:
            for var in local_model.varlist:
                curr = getattr(local_model.path,var)
                setattr(local_model.path,var,curr[0,:])
        models[HH] = local_model
        compiled = True
    del model_
    print(f'Finished!')
    return models


def sol_models_sensitivity(parname, parvals, upars_HH=None, HH='HA', scalevar='C_s', cumeffect=False, size=0.01, do_reduction=False, scale_kwargs=None):
     
    upars_foreign = {}
    model_foreign = GetForeignEcon.get_foreign_econ(shocksize=-0.001, upars=upars_foreign)


    models = {}
    compiled = False
    
 
    for parval in parvals: 
        print(parval)
        #models[par][HH] = {}
        if compiled:
            pass
        else:
            model_ = HANKModelClass(name=HH)
            orgpar = deepcopy(model_.par)
        model_.par = deepcopy(orgpar)
        
        model_.par.HH_type = HH
        #for param in upars:
        setattr(model_.par,parname,parval)
        if upars_HH is not None:
            for param in upars_HH[HH]:
                setattr(model_.par,param,upars_HH[HH][param])

        model_.find_ss()
        if model_.par.HH_type == 'HA':
            model_.compute_jacs(do_print=False,skip_shocks=True)
        else:
            model_.compute_jacs(do_print=False,skip_shocks=True,skip_hh=True)
        GetForeignEcon.create_foreign_shock(model_, model_foreign)
        model_.transition_path(do_print=False)
        if scale_kwargs is None:
            utils.scaleshock(scalevar, model_, size=size, cumeffect=cumeffect)
        else:
            utils.scaleshock(model=model_, **scale_kwargs)   
        local_model = deepcopy(model_)
        del local_model.jac 
        del local_model.sim 
        del local_model.sol 
        del local_model.jac_hh
        if do_reduction:
            for var in local_model.varlist:
                curr = getattr(local_model.path,var)
                setattr(local_model.path,var,curr[0,:])
        models[parval] = local_model
        compiled = True
   
    del model_
    print(f'Finished!')
    return models
   
