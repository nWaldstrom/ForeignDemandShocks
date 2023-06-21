# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 15:04:05 2021

@author: fcv495
"""

#%autoreload


import os
import sys
sys.path.append("..")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import time
import numpy as np

import matplotlib.pyplot as plt   
plt.style.use('seaborn-whitegrid')
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

from ForeignEcon import NKModelClass
# import utils
# import comp
# import figs


#%%

model = NKModelClass(name='baseline')

#from Foreign_steady_state import find_foreign_ss

model.find_ss(do_print=True)



#%%


model.compute_jacs(do_print=True,skip_shocks=True,skip_hh=True)

#%%



dGamma = np.zeros(model.par.T)
dGamma[:5] = -0.0001
shock_specs = {'betaF':dGamma}

model.par.max_iter_broyden = 20
# b. find transition path
model.find_transition_path(shock_specs=shock_specs, do_print=True)

#%%
model.show_IRFs(['C_s','PF_s'],T_max=40,do_shocks=False,do_targets=False)




#%%
# if model.par.deltaK > 0:
#     model.par.phiK = 0.08/model.par.deltaK 
#phiK = 8
#model.par.phiK = 1/(model.par.deltaK * phiK)

# model.par.ModelForeignEcon = True 
# model.par.floating = True

if md.par.HH_type == 'HA':
    md.compute_jac_hh()



# model.compute_jac(do_print=False)
md.compute_jac(do_print=False, parallel=False)

# comp.compute_jac_exo(model)
# model.check_determinancy()


# utils.compute_jac_sym(model, h=1e-04)
# utils.compute_jac_exo_sym(model, h=1e-04)
# utils.Create_shock('UniformT', md, 0.0001, absval=True)
utils.Create_shock('betaF', md, -0.0001, absval=False)
# utils.Create_shock('G_eps', md, 0.001, absval=False)

# utils.Create_shock('SupplySF, model, 0.001)
md.par.max_iter_broyden = 30


# md.path.G_eps[0,:] = md.ss.G_eps 
# md.path.G_eps[0,40:80] = md.ss.G_eps + 0.001 * md.ss.G_eps * 0.5**(np.arange(40))

# md.find_transition_path(do_print=True)
comp.find_transition_path(md, do_print=True, use_jac_hh=False)

# comp.find_transition_path_scipy(model, do_print=True)


#comp.T_path_using_init_jac(model, do_print=True)

# utils.scaleshock('G_eps', md)
utils.scaleshock('C_s', md)

abs_value = ['r', 'ra', 'iF_s', 'goods_mkt', 'NKPC', 'Taylor_rule', 'StockPrice', 'E', 'NFA', 'Walras', 'piNT', 'iF_s', 'pi', 'piH', 'piF', 'UniformT']
#paths = ['C_s','iF_s','piF']  
#figs.show_IRFs_new([model], paths,abs_value=abs_value,T_max=40, shocktitle='Demand shock', scale=True, do_sumplot=False)
paths = ['Y','C','r', 'ra', 'Walras', 'A',  'Q', 'pi', 'piNT', 'piH', 'wnT', 'wnNT', 'YT', 'YNT', 'NT', 'NNT', 'B', 'LT', 'G', 'YT'] 
#paths = ['Y','C','r', 'ra', 'Walras', 'A',  'Q', 'pi', 'piNT', 'piH', 'wnT', 'wnNT', 'YT', 'YNT', 'NT', 'NNT', 'B', 'LT', 'G', 'E', 'I', 'UniformT', 'IT', 'INT', 'rkT', 'rkNT', 'C_s', 'betaF', 'PF_s', 'iF_s', 'i'] 
figs.show_IRFs_new([md], paths,abs_value=abs_value,T_max=300, shocktitle='Demand shock', do_sumplot=False, scale=True)


#%%
paths = ['C_s','iF_s','piF'] 
figs.show_IRFs_new([md], paths,abs_value=abs_value,T_max=300, shocktitle='Demand shock', do_sumplot=False, scale=True)

# figs.C_decomp(model=model, var='C', T_max=50)
# figs.C_decomp(model=model, var='C_T', T_max=50)
# figs.C_decomp(model=model, var='C_NT', T_max=50)


#%%
paths = ['G', 'B', 'LT'] 
figs.show_IRFs_new([md], paths,abs_value=abs_value,T_max=400, shocktitle='Demand shock', scale=True, do_sumplot=False)

#%%
pathlabels = ['G', 'Lumpsum taxes', 'Gov. Bonds', 'Output', 'Consumption', 'r', 'CPI Inflation', 'Real Exchange rate']
paths = ['G', 'LT', 'B', 'Y', 'C', 'r', 'pi', 'Q'] 
pctp = ['r', 'pi']
abs_value = []
colors = ['Darkgreen']*len(paths)
figs.show_IRFs_new([md], paths,abs_value=abs_value,T_max=100, pctp=pctp,  scale=True, do_sumplot=False, pathlabels=pathlabels, colors=colors)
# fig.savefig('plots/G_shock.png')



#%%
pathlabels = ['G', 'Lumpsum taxes', 'Gov. Bonds', 'Output', 'Consumption', 'r', 'CPI Inflation', 'Real Exchange rate']
paths = ['G', 'LT', 'B', 'Y', 'C', 'r', 'pi', 'Q'] 
pctp = ['r', 'pi']
abs_value = []
colors = ['Darkgreen']*len(paths)
fig=figs.show_IRFs_new([md], paths,abs_value=abs_value,T_max=100, pctp=pctp,  scale=True, do_sumplot=False, pathlabels=pathlabels, colors=colors)
fig.savefig('plots/G_shock_no_repayment.png')


#%%
pathlabels = ['G', 'Lumpsum taxes', 'Gov. Bonds', 'Output', 'Consumption', 'r', 'CPI Inflation', 'Real Exchange rate']
paths = ['G', 'LT', 'B', 'Y', 'C', 'r', 'pi', 'Q'] 
pctp = ['r', 'pi']
abs_value = []
colors = ['Darkgreen']*len(paths)
fig=figs.show_IRFs_new([md], paths,abs_value=abs_value,T_max=100, pctp=pctp,  scale=False, do_sumplot=False, pathlabels=pathlabels, colors=colors)
fig.savefig('plots/T_announcement.png')




#%%


# varlist = ['Y','YT','YNT', 'C']
# dXs = figs.vary_irfs(shock='betaF', varlist=varlist,scalevar='C_s', paramlist='etaT_LR', paramvals=[0.5, 3], HH_type='TA-IM', jump=0.0001, T_max=40)

# varlist = ['Y','YT','YNT', 'C', 'C_s']
# dXs = figs.vary_irfs(shock='betaF', varlist=varlist,scalevar='C_s', paramlist='etaT_LR', paramvals=[0.5, 1.1, 3], HH_type='HA', jump=0.0001, T_max=40)

# varlist = ['Y','YT','YNT', 'C', 'C_s']
# dXs = figs.vary_irfs(shock='betaF', varlist=varlist,scalevar='C_s', paramlist='theta_T_NT', paramvals=[0.0, 0.5, 0.8], HH_type='RA-IM', jump=0.0001, T_max=40)

# varlist = ['Y','YT','YNT', 'C', 'C_s']
# dXs = figs.vary_irfs(shock='betaF', varlist=varlist,scalevar='C_s', paramlist='theta_T_NT', paramvals=[0.0, 0.5, 0.8], HH_type='HA', jump=0.0001, T_max=40)


# varlist = ['Y','YT','YNT', 'C', 'r', 'pi', 'B', 'LT']
# dXs = figs.vary_irfs(shock='G_eps', varlist=varlist,scalevar='G_eps', paramlist='deltaB', paramvals=[30, 50, 70], HH_type='TA-IM', jump=0.0001, T_max=100)


# varlist = ['Y','YT','YNT', 'C', 'r', 'pi', 'B', 'LT']
# dXs = figs.vary_irfs(shock='G_eps', varlist=varlist,scalevar='G_eps', paramlist='epsB', paramvals=[0.001, 0.005, 0.01, 0.03], HH_type='TA-IM', jump=0.0001, T_max=250)

varlist = ['Y','YT','YNT', 'C', 'r', 'pi', 'B', 'LT']
dXs = figs.vary_irfs(shock='G_eps', varlist=varlist,scalevar='G_eps', paramlist='epsB', paramvals=[ 0.005], HH_type='TA-IM', jump=0.0001, T_max=250)



#%%
shock='betaF'
varlist = ['Y','YT','YNT', 'C', 'C_s']
scalevar='C_s'
paramlist='r_debt_elasticity'
paramvals=[0.0]
HH_type='TA-IM'
jump=0.0001
T_max=40

N_models = len(paramvals)
if isinstance(varlist,list):
    N_vars = len(varlist)
else:
    N_vars = 1
    varlist = [varlist]
models = [HANKModelClass(name='vers'+str(ii)) for ii in range(N_models)]
dXs = np.zeros((N_models,N_vars,models[0].par.transition_T))

for ii in range(N_models):
    model_ = models[ii]
    # model_.par.HH_type = HH_type
    setattr(model_.par, 'HH_type', HH_type)
    if isinstance(paramlist, list):
        for param in paramlist:
            setattr(model_.par, param, paramvals[ii])
        paramstr = ' = '.join(paramlist)
    else:
        setattr(model_.par, paramlist, paramvals[ii])
        paramstr = paramlist
    print(f'{paramstr} = {paramvals[ii]:.2f}')

    # Do stuff
    model_.find_ss(do_print=False)
    if model_.par.HH_type=='HA':
        model_.compute_jac_hh(do_print=False)
    model_.compute_jac(do_print=False, parallel=False)
    utils.Create_shock('betaF', model_, 0.0001)

    
    model_.find_transition_path(do_print=True)
    utils.scaleshock(scalevar, model_)
    scale = getattr(model_.par,'scale')
    print('\n')

    # Get IRFs
    for jj in range(len(varlist)):
        dXs[ii,jj,:] = utils.get_dX(varlist[jj], model_, scaleval=scale)  




#%%


num = len(varlist) 
nrows = num//4+1
ncols = np.fmin(num,4)
if num%4 == 0: nrows -= 1 
fig = plt.figure(figsize=(4*ncols,3*nrows))
# fig,ax = plt.figure(figsize=(4*ncols,4*nrows))

   
# fig,ax = plt.subplots(1,N_vars,figsize=(4*N_vars,3))
alphalist = np.flip(np.linspace(0.4,1,N_models))
col = 'Darkgreen'
for jj in range(N_vars):
    ax = fig.add_subplot(nrows,ncols,jj+1)
    for ii in range(N_models):
        ax.plot(dXs[ii,jj,:T_max]*100,label=f'{paramstr} = {paramvals[ii]:.2f}',
                    alpha=alphalist[ii], linewidth=2, color=col)
    ax.set_ylabel('% diff. to s.s.')
    ax.set_title(varlist[jj])
plt.gca().legend()
fig.tight_layout(pad=1.6)


#%%

paths = ['Y','C','r', 'ra', 'Walras', 'A',  'Q', 'pi', 'piNT', 'piH', 'wnT', 'wnNT', 'YT', 'YNT', 'NT', 'NNT', 'B', 'LT', 'G', 'E', 'I', 'K', 'IT', 'INT', 'rkT', 'rkNT', 'C_s', 'betaF', 'PF_s', 'iF_s'] 
figs.show_IRFs_new([model_], paths,abs_value=abs_value,T_max=40, shocktitle='Demand shock', scale=True, do_sumplot=False)

#%%

var = 'K'
test = (getattr(md.path,var)[0,:600]/getattr(md.ss,var)-1) * 100
test1 =  (getattr(model_.path,var)[0,:600]/getattr(model_.ss,var)-1) * 100
plt.plot(test)
plt.plot(test1,'--')
plt.show()


#%%

var = 'A'
test = (getattr(md.path,var)[0,:600]/getattr(md.ss,var)-1) * 100
test1 =  (getattr(model_.path,var)[0,:600]/getattr(model_.ss,var)-1) * 100
pltest = test - test1
plt.plot(pltest)
plt.show()



#%%

test = np.zeros(400)

for t in range(400):
    if t >= 40:
        w = 0.001
        sigmaD = 1.0 / (1.0 +w* np.exp((t-40)*0.15) )
        x =  model.par.etaT_SR + (model.par.etaT_LR - model.par.etaT_SR)*(1 - sigmaD)
        test[t] = x
    else:
        test[t] = model.par.etaT_SR

plt.plot(np.arange(400), test)
plt.show()



#%%

# utils.compute_jac_sym(model, h=1e-04)
# utils.compute_jac_exo_sym(model, h=1e-04)
utils.Create_shock('G_eps', model, 0.001)
# utils.Create_shock('SupplySF, model, 0.001)
model.par.max_iter_broyden = 20



# model.find_transition_path(do_print=True)
comp.find_transition_path(model, do_print=True, use_jac_hh=True)
# utils.T_path_using_init_jac(model, do_print=True)

utils.scaleshock('G_eps', model)

abs_value = ['r', 'ra', 'iF_s', 'goods_mkt', 'NKPC', 'Taylor_rule', 'StockPrice', 'E', 'NFA', 'Walras', 'piNT', 'iF_s', 'pi', 'piH', 'piF']
#paths = ['C_s','iF_s','piF'] 
#figs.show_IRFs_new([model], paths,abs_value=abs_value,T_max=40, shocktitle='Demand shock', scale=True, do_sumplot=False)

paths = ['Y','C','r', 'ra', 'Walras', 'A',  'Q', 'pi', 'piNT', 'piH', 'wnT', 'wnNT', 'YT', 'YNT', 'NT', 'NNT', 'B', 'LT', 'G', 'E', 'r'] 
figs.show_IRFs_new([model], paths,abs_value=abs_value,T_max=50, shocktitle='Demand shock', scale=True, do_sumplot=False)


#%%


# utils.compute_jac_sym(model, h=1e-04)
# utils.compute_jac_exo_sym(model, h=1e-04)
utils.Create_shock('G_eps', model, 0.001)
# utils.Create_shock('SupplySF', model, 0.001)
model.par.max_iter_broyden = 20


# model.find_transition_path(do_print=True)
comp.find_transition_path(model, do_print=True, use_jac_hh=True)
# utils.T_path_using_init_jac(model, do_print=True)

utils.scaleshock('G_eps', model)


abs_value = ['r', 'ra', 'iF_s', 'goods_mkt', 'NKPC', 'Taylor_rule', 'StockPrice', 'E', 'NFA', 'Walras', 'piNT', 'iF_s', 'pi', 'piH', 'piF']
#paths = ['C_s','iF_s','piF'] 
#figs.show_IRFs_new([model], paths,abs_value=abs_value,T_max=40, shocktitle='Demand shock', scale=True, do_sumplot=False)

paths = ['Y','C','r', 'ra', 'Walras', 'A',  'Q', 'pi', 'piNT', 'piH', 'wnT', 'wnNT', 'YT', 'YNT', 'NT', 'NNT', 'B', 'LT', 'G', 'NFA'] 
figs.show_IRFs_new([model], paths,abs_value=abs_value,T_max=40, shocktitle='Demand shock', scale=True, do_sumplot=False)




#%%

paths = ['C_s','piF', 'PF_s', 'iF_s' ] 


abs_value = ['r', 'ra', 'iF_s', 'goods_mkt', 'NKPC', 'Taylor_rule', 'StockPrice', 'E', 'NFA', 'Walras', 'piNT', 'iF_s', 'pi', 'piH']
figs.show_IRFs_new([model], paths,abs_value=abs_value,T_max=400, shocktitle='Demand shock', scale=True, do_sumplot=True)





#%%

def vary_irfs(shock,varlist,scalevar,paramlist,paramvals,HH_type='HA',jump=0.001,T_max=40):
    ''' IRFs of different variables to a shock for different parameter values
    
    Parameters
    ----------
    shock:     string, variable to shock
    varlist:   string or list, variable(s) to shock
    scalevar:  variable that shock is scaled according to
    paramlist: string or list, parameter(s) to vary
    paramvals: object, values that parameters take
    jump:      float, size of shock
    T_max:     int, number of time periods to plot
    
    Returns
    ----------
    dXs:       array, impulse response functions (nModels,nVars,T)
    
    '''
    
    N_models = len(paramvals)
    if isinstance(varlist,list):
        N_vars = len(varlist)
    else:
        N_vars = 1
        varlist = [varlist]
    models = [HANKModelClass() for i in range(N_models)]
    dXs = np.zeros((N_models,N_vars,models[0].par.transition_T))
    
    for i in range(N_models):
        model = models[i]
        model.par.HH_type = HH_type
        if isinstance(paramlist, list):
            for param in paramlist:
                setattr(model.par, param, paramvals[i])
            paramstr = ' = '.join(paramlist)
        else:
            setattr(model.par, paramlist, paramvals[i])
            paramstr = paramlist
        print(f'{paramstr} = {paramvals[i]:.2f}')

        # Do stuff
        model.find_ss(do_print=False)
        model.par.ModelForeignEcon = True 
        if HH_type=='HA':
            model.compute_jac_hh(do_print=False)
        model.compute_jac(do_print=False)
        utils.Create_shock(shock, model, jump)
        model.find_transition_path(do_print=False)
        utils.scaleshock(scalevar, model)
        scale = getattr(model.par,'scale')
        print('\n')

        # Get IRFs
        for j in range(len(varlist)):
            dXs[i,j,:] = utils.get_dX(varlist[j], model, scaleval=scale)
            
    fig,ax = plt.subplots(1,N_vars,figsize=(4*N_vars,3))
    for j in range(N_vars):
        for i in range(N_models):
            ax[j].plot(dXs[i,j,:T_max]*100,label=f'{paramstr} = {paramvals[i]:.2f}')
        ax[j].set_title(varlist[j])
    plt.gca().legend()
        
    return dXs




#jump = 0.001
jump = -0.001
scalevar = 'C_s'
T_max = 10
varlist = ['YNT','YT','Y', 'C_T', 'C_NT']

dXs = vary_irfs('betaF',varlist,scalevar,'floating',[True, False])





#%% Jac check - not exact!

# comp.compute_jac_exo(model)
# model.compute_jac(do_print=False)
# comp.compute_jac_exo(model)
    

def get_G_mat(model):
    T = model.par.transition_T
    N_endo, N_exo = len(model.inputs_endo), len(model.inputs_exo)
    temp = np.reshape(model.jac_exo[:,:T,:], (T*N_endo,T*N_exo))
    G = -np.linalg.solve(model.jac, temp)
    G = np.reshape(G, (T*N_endo,T,N_exo))
    return G  

dBetaF = 0.01* 0.9**np.arange(model.par.transition_T)

vn = 7
vname = 'YT'
Tplot = 300
G = get_G_mat(model)

dx = G[vn*model.par.transition_T:(1+vn)*model.par.transition_T,:,6] @ dBetaF
#dE = getattr(model.path,vname)[0,:Tplot] - getattr(model.ss,vname)


#plt.plot(dE)
plt.plot(dx[:50], '-')
plt.show( )


#%%

dwnT = model.path.wnT[0,:] #- ss.wnT
dwnNT = model.path.wnNT[0,:]  #- ss.wnT
dra = model.path.ra[0,:] 

a_c = model.path.A[0,:] + model.path.C[0,:]
A_lag = np.zeros(400)
A_lag[1:] = model.path.A[0,:-1]
A_lag[0] = model.ss.A 


m = (1+dra)*A_lag + model.par.sT*dwnT + (1-model.par.sT)*dwnNT

C = model.path.C[0,:]
A = model.path.A[0,:] 
I = model.par.sT*dwnT + (1-model.par.sT)*dwnNT
plt.plot(m-a_c)
plt.show()

res = np.zeros(400)
for t in range(model.par.transition_T):  
     if t==0:
         res[t] = A[t] - (model.ss.A * (1+dra[t]) + I[t]   - C[t] )
     else:
         res[t] = A[t] - (A_lag[t] * (1+dra[t]) + I[t]   - C[t] )

plt.plot(res)
plt.show( )



#%%

# model.par.HH_type = 'HA'

HAmodel = HANKModelClass(name='HA')
xinit = np.array([0.993, 1.004, 0.92, 1.14])
HAmodel.par.x0 = xinit
HAmodel.UseFixed_r()
HAmodel.find_ss(do_print=True)
HAmodel.compute_jac_hh(do_print=False, do_simple=False)

ss = model.ss 

dr = model.path.r[0,:] 
dra = model.path.ra[0,:] 
dwnT = model.path.wnT[0,:] 
dwnNT = model.path.wnNT[0,:] 


dA = HAmodel.jac_hh.A_r@(dr-ss.r) 
dA[:] += HAmodel.jac_hh.A_ra@(dra-ss.ra) 
dA[:] += HAmodel.jac_hh.A_wnT@(dwnT-ss.wnT) 
dA[:] += HAmodel.jac_hh.A_wnNT@(dwnNT-ss.wnNT) 
          
dA_full = model.path.A[0,:] - model.ss.A
   
plt.plot(dA_full)     
plt.plot(dA,'--')
plt.show()

dC = HAmodel.jac_hh.C_r@(dr-ss.r) 
dC[:] += HAmodel.jac_hh.C_ra@(dra-ss.ra) 
dC[:] += HAmodel.jac_hh.C_wnT@(dwnT-ss.wnT) 
dC[:] += HAmodel.jac_hh.C_wnNT@(dwnNT-ss.wnNT) 
          
dC_full = model.path.C[0,:] - model.ss.C
   
plt.plot(dC_full)     
plt.plot(dC,'--')
plt.show()

dC_NT = HAmodel.jac_hh.C_NT_r@(dr-ss.r) 
dC_NT[:] += HAmodel.jac_hh.C_NT_ra@(dra-ss.ra) 
dC_NT[:] += HAmodel.jac_hh.C_NT_wnT@(dwnT-ss.wnT) 
dC_NT[:] += HAmodel.jac_hh.C_NT_wnNT@(dwnNT-ss.wnNT) 
          
dC_NT_full = model.path.C_NT[0,:] - model.ss.C_NT
   
plt.plot(dC_NT_full)  
plt.plot(dC_NT,'--')
plt.show()


dC_T = HAmodel.jac_hh.C_T_r@(dr-ss.r) 
dC_T[:] += HAmodel.jac_hh.C_T_ra@(dra-ss.ra) 
dC_T[:] += HAmodel.jac_hh.C_T_wnT@(dwnT-ss.wnT) 
dC_T[:] += HAmodel.jac_hh.C_T_wnNT@(dwnNT-ss.wnNT) 
          
dC_T_full = model.path.C_T[0,:] - model.ss.C_T
plt.plot(dC_T_full)  
plt.plot(dC_T,'--')
plt.show()




#%%

test = G[vn*model.par.transition_T:(1+vn)*model.par.transition_T,:,0]


plt.plot(model.jac_exo[600,[350,800],0]*100)
plt.show()




#%%

paths = ['UCT','UCNT']
abs_value = ['r', 'ra', 'iF_s', 'goods_mkt', 'NKPC', 'Taylor_rule', 'StockPrice', 'E', 'NFA', 'Walras']
figs.show_IRFs_new([model], paths,abs_value=abs_value,T_max=200, shocktitle='Exchange rate shock', scale=True, do_sumplot=True)



#%% Foreign demand shock 

model.par.max_iter_broyden = 10
utils.Create_shock('C_s', model)
#model.find_transition_path(do_print=True)
comp.T_path_using_init_jac(model, do_print=True)
utils.scaleshock('C_s', model)

paths = ['Y','C','r','wnNT', 'wnT',  'NFA', 'Q', 'Walras', 'C_s']
abs_value = ['r', 'goods_mkt', 'NKPC', 'Taylor_rule', 'StockPrice', 'E', 'NFA', 'Walras']
figs.show_IRFs_new([model], paths,abs_value=abs_value,T_max=40, shocktitle='Foreign demand shock', scale=False, do_sumplot=True)


#%%
# Q shock 
model.par.HH_type = 'RA'
model.find_ss(do_print=True)

model.compute_jac(do_print=True, parallel=False, use_jac_hh=False)

model.par.max_iter_broyden = 10
model.par.jump_Z = 0.0
model.par.jump_iF_s = 0.01  
model.find_transition_path(do_print=True)

dQ_init = model.path.Q[0,0]/model.ss.Q-1
model.par.scale = 0.01 / dQ_init

paths = ['Y','C','r','wnNT', 'wnT',  'NFA', 'Q', 'Walras']
abs_value = ['r', 'goods_mkt', 'NKPC', 'Taylor_rule', 'StockPrice', 'E', 'NFA', 'Walras']
figs.show_IRFs_new([model], paths,abs_value=abs_value,T_max=100, shocktitle='TPF Shock', scale=True, do_sumplot=True)



#%% Compare HA, RA, TA

model_HA_hetsec = model.copy(name='HA het sec')
model_HA_hetsec.par.HH_type = 'HA'

# model_HA_monosec = model.copy(name='HA mono sec')
# model_HA_monosec.par.HH_type = 'HA'
# model_HA_monosec.MonoSec() 


model_RA = model.copy(name='RA-CM')
model_RA.par.HH_type = 'RA-CM'

model_TA = model.copy(name='TA-CM')
model_TA.par.HH_type = 'TA-CM'

models = [model_HA_hetsec, model_RA, model_TA]
 
for model_ in models:   
    print(f'###')
    print(f'### {model_.name}')
    print(f'###\n')
    
    model_.find_ss(do_print=False)
    print('')
    model_.compute_jac_hh(do_print=False)
    model_.compute_jac(do_print=False)
    print('')
    model_.par.jump_C_s = 0.01
    model_.par.jump_iF_s = 0.0
    model._set_inputs_exo()
    model_.par.max_iter_broyden = 40
    model_.find_transition_path(do_print=True)
    print('')
    

paths = ['Y','C','wnT']
labels = [r'HA het sec', r'RA het sec', r'TA het sec']

figs.show_IRFs_new(models=models, paths=paths,labels=labels,
                    T_max=40, do_sumplot=False, scale=False, ldash=['-', '-', '-.', '--'])    



#%% Compare with different MPCs

# Same MCPCs across sectors 
model_HA_mono_MPC = HANKModelClass(name='HA mono MPC')
model_HA_mono_MPC.UseFixed_r()
model_HA_mono_MPC.HH_type = 'HA'
model_HA_mono_MPC.par.x0 = xinit
model_HA_mono_MPC.find_ss(do_print=True)
dMPC = 0.2

# Higher MPCs in tradeable sector 
model_HA_high_MPCT = HANKModelClass(name='High MPC T')
model_HA_high_MPCT.UseFixed_r()
model_HA_high_MPCT.HH_type = 'HA'
MPC_T_target = model_HA_mono_MPC.par.Agg_MPC + dMPC
MPC_NT_target  = (model_HA_high_MPCT.par.Agg_MPC - model_HA_high_MPCT.par.sT * MPC_T_target)/(1-model_HA_high_MPCT.par.sT)


# Homotopy continuation
model_HA_high_MPCT.par.x0 = xinit
Nit = 10 
for x in range(Nit):
    k = (1+x) / Nit
    model_HA_high_MPCT.par.Agg_MPC_T = model_HA_high_MPCT.par.Agg_MPC * (1-k) + k * MPC_T_target
    model_HA_high_MPCT.par.Agg_MPC_NT = model_HA_high_MPCT.par.Agg_MPC * (1-k) + k * MPC_NT_target
    model_HA_high_MPCT.find_ss(do_print=False)
    if np.isclose(k,1):
        print(f'Final iteration')  
  

# Lower MPCs in tradeable sector 
model_HA_low_MPCT = HANKModelClass(name='Low MPC T')
model_HA_low_MPCT.UseFixed_r()
model_HA_low_MPCT.HH_type = 'HA'
MPC_T_target = model_HA_mono_MPC.par.Agg_MPC - dMPC
MPC_NT_target  = (model_HA_low_MPCT.par.Agg_MPC - model_HA_low_MPCT.par.sT * MPC_T_target)/(1-model_HA_low_MPCT.par.sT)

# Homotopy continuation
model_HA_low_MPCT.par.x0 = xinit
Nit = 10 
for x in range(Nit):
    k = (1+x) / Nit
    model_HA_low_MPCT.par.Agg_MPC_T = model_HA_low_MPCT.par.Agg_MPC * (1-k) + k * MPC_T_target
    model_HA_low_MPCT.par.Agg_MPC_NT = model_HA_low_MPCT.par.Agg_MPC * (1-k) + k * MPC_NT_target
    model_HA_low_MPCT.find_ss(do_print=False)
    if np.isclose(k,1):
        print(f'Final iteration')
        
    

models = [model_HA_mono_MPC, model_HA_high_MPCT, model_HA_low_MPCT]

shocktypes = ['ForeignDemand', 'ExchangeRate']
shock = shocktypes[0]

for model_ in models:   
    model_.par.gamma  = model_.par.eta = model_.par.etaT =  3.0

    print(f'###')
    print(f'### {model_.name}')
    print(f'###\n')
    
    print('')
    model_.compute_jac_hh(do_print=False)
    model_.compute_jac(do_print=False)
    print('')
    if shock == 'ForeignDemand':
        model_.par.jump_C_s = 0.01
        model_.par.jump_iF_s = 0.0
    elif shock == 'ExchangeRate':
        model_.par.jump_C_s = 0.0
        model_.par.jump_iF_s = 0.01
    model_._set_inputs_exo()
    model_.par.max_iter_broyden = 80
    model_.find_transition_path(do_print=True)
    print('')
    #dY_init = model_.path.Y[0,0]/model_.ss.Y-1
    #model_.par.scale = -0.01 / dY_init
    
    #dQ_init = model_.path.C_s[0,0]/model_.ss.C_s-1
    #model_.par.scale = 0.01 / dQ_init
    if shock == 'ForeignDemand':
        dExo_init = model_.path.C_s[0,0]/model_.ss.C_s-1
    elif shock == 'ExchangeRate':
        dExo_init = model_.path.Q[0,0]/model_.ss.Q-1
    model_.par.scale = 0.01 / dExo_init


paths = ['Y','C','C_T', 'C_NT', 'wnT', 'wnNT', 'YT', 'YNT']
pathlabels = ['Output', 'Agg. C', 'Avg. C in tradeable sec.', 'Avg. C in Non-tradeable sec.',
              'Real income T', 'Real income NT', 'Tradeable Output', 'Non-tradeable Output']

labels = [r'$MPC_T = MPC_{NT}$' +  f' = {model_HA_mono_MPC.par.Agg_MPC:4.2f}', 
          f'$MPC_T$ = {model_HA_high_MPCT.par.Agg_MPC_T:4.2f}, ' + r'$MPC_{NT}$' + f' = {model_HA_high_MPCT.par.Agg_MPC_NT:4.2f}',
          f'$MPC_T$ = {model_HA_low_MPCT.par.Agg_MPC_T:4.2f}, ' + r'$MPC_{NT}$' + f' = {model_HA_low_MPCT.par.Agg_MPC_NT:4.2f}']
colors=['Blue', 'firebrick', 'darkgreen']

fig = figs.show_IRFs_new(models=models, paths=paths,labels=labels, pathlabels=pathlabels,
                    T_max=40, do_sumplot=False, scale=True, 
                    ldash=['-', '--', '-.'], colors=colors, lwidth=2)   
if shock == 'ForeignDemand': 
    fig.savefig('plots\Varying_MPCs_dCs_high_xi.pdf')
elif shock == 'ExchangeRate':
    fig.savefig('plots\Varying_MPCs_dQ_xi.pdf')

for model_ in models:
    del model_ 


#%% Compare with different MPCs - Taylor Rule

# Same MCPCs across sectors 
model_HA_mono_MPC = HANKModelClass(name='HA mono MPC')
model_HA_mono_MPC.HH_type = 'HA'
model_HA_mono_MPC.UseTaylor() 
model_HA_mono_MPC.par.x0 = xinit
model_HA_mono_MPC.find_ss(do_print=True)
dMPC = 0.1

# Higher MPCs in tradeable sector 
model_HA_high_MPCT = HANKModelClass(name='High MPC T')
model_HA_high_MPCT.HH_type = 'HA'
model_HA_high_MPCT.UseTaylor() 
MPC_T_target = model_HA_mono_MPC.par.Agg_MPC + dMPC
MPC_NT_target  = (model_HA_high_MPCT.par.Agg_MPC - model_HA_high_MPCT.par.sT * MPC_T_target)/(1-model_HA_high_MPCT.par.sT)


# Homotopy continuation
model_HA_high_MPCT.par.x0 = xinit
Nit = 10 
for x in range(Nit):
    k = (1+x) / Nit
    model_HA_high_MPCT.par.Agg_MPC_T = model_HA_high_MPCT.par.Agg_MPC * (1-k) + k * MPC_T_target
    model_HA_high_MPCT.par.Agg_MPC_NT = model_HA_high_MPCT.par.Agg_MPC * (1-k) + k * MPC_NT_target
    model_HA_high_MPCT.find_ss(do_print=False)
    if np.isclose(k,1):
        print(f'Final iteration')  
  

# Lower MPCs in tradeable sector 
model_HA_low_MPCT = HANKModelClass(name='Low MPC T')
model_HA_low_MPCT.HH_type = 'HA'
model_HA_low_MPCT.UseTaylor() 
MPC_T_target = model_HA_mono_MPC.par.Agg_MPC - dMPC
MPC_NT_target  = (model_HA_low_MPCT.par.Agg_MPC - model_HA_low_MPCT.par.sT * MPC_T_target)/(1-model_HA_low_MPCT.par.sT)

# Homotopy continuation
model_HA_low_MPCT.par.x0 = xinit
Nit = 10 
for x in range(Nit):
    k = (1+x) / Nit
    model_HA_low_MPCT.par.Agg_MPC_T = model_HA_low_MPCT.par.Agg_MPC * (1-k) + k * MPC_T_target
    model_HA_low_MPCT.par.Agg_MPC_NT = model_HA_low_MPCT.par.Agg_MPC * (1-k) + k * MPC_NT_target
    model_HA_low_MPCT.find_ss(do_print=False)
    if np.isclose(k,1):
        print(f'Final iteration')
        
#%%
models = [model_HA_mono_MPC, model_HA_high_MPCT, model_HA_low_MPCT]

for model_ in models:   
    print(f'###')
    print(f'### {model_.name}')
    print(f'###\n')
    
    print('')
    model_.compute_jac_hh(do_print=False)
    model_.compute_jac(do_print=False)
    print('')
    model_.par.jump_C_s = 0.0
    model_.par.jump_iF_s = 0.01
    model_._set_inputs_exo()
    model_.par.max_iter_broyden = 50
    model_.find_transition_path(do_print=True)
    print('')
    #dY_init = model_.path.Y[0,0]/model_.ss.Y-1
    #model_.par.scale = -0.01 / dY_init
    
    #dQ_init = model_.path.C_s[0,0]/model_.ss.C_s-1
    #model_.par.scale = 0.01 / dQ_init
    dQ_init = model_.path.Q[0,0]/model_.ss.Q-1
    model_.par.scale = 0.01 / dQ_init

paths = ['Y','C','C_T', 'C_NT', 'wnT', 'wnNT', 'YT', 'YNT']
pathlabels = ['Output', 'Agg. C', 'Avg. C in tradeable sec.', 'Avg. C in Non-tradeable sec.',
              'Real income T', 'Real income NT', 'Tradeable Output', 'Non-tradeable Output']

labels = [r'$MPC_T = MPC_{NT}$' +  f' = {model_HA_mono_MPC.par.Agg_MPC:4.2f}', 
          f'$MPC_T$ = {model_HA_high_MPCT.par.Agg_MPC_T:4.2f}, ' + r'$MPC_{NT}$' + f' = {model_HA_high_MPCT.par.Agg_MPC_NT:4.2f}',
          f'$MPC_T$ = {model_HA_low_MPCT.par.Agg_MPC_T:4.2f}, ' + r'$MPC_{NT}$' + f' = {model_HA_low_MPCT.par.Agg_MPC_NT:4.2f}']
colors=['Blue', 'firebrick', 'darkgreen']

fig = figs.show_IRFs_new(models=models, paths=paths,labels=labels, pathlabels=pathlabels,
                    T_max=40, do_sumplot=False, scale=True, 
                    ldash=['-', '--', '-.'], colors=colors, lwidth=2)    
#fig.savefig('plots\Varying_MPCs_dCs_taylor.pdf')
fig.savefig('plots\Varying_MPCs_dQ_taylor.pdf')

for model_ in models:
    del model_ 


#%% Compare with endo. uncertainty 

# Exo. uncertainty 
model_HA_no_uncertainty = HANKModelClass(name='No endo uncertainty')
model_HA_no_uncertainty.par.x0 = xinit
model_HA_no_uncertainty.find_ss(do_print=True)

# Some endo. uncertainty 
model_HA_uncerntainty_T_NT = model_HA_no_uncertainty.copy(name='uncertainty_T_NT')
model_HA_uncerntainty_T_NT.par.el_Ye_T = 0.1
model_HA_uncerntainty_T_NT.par.el_Ye_NT = 0.1

# large endo. uncertainty 
model_HA_uncerntainty_T_NT_large = model_HA_no_uncertainty.copy(name='uncertainty_T_NT_large')
model_HA_uncerntainty_T_NT_large.par.el_Ye_T = 1
model_HA_uncerntainty_T_NT_large.par.el_Ye_NT = 1

models = [model_HA_no_uncertainty, model_HA_uncerntainty_T_NT, model_HA_uncerntainty_T_NT_large]

for model_ in models:   
    print(f'###')
    print(f'### {model_.name}')
    print(f'###\n')
    
    #model_.find_ss(do_print=True)
    print('')
    model_.compute_jac_hh(do_print=False)
    model_.compute_jac(do_print=False)
    print('')
    model_.par.jump_iF_s = 0.01
    model_.par.jump_C_s = 0.00
    model_._set_inputs_exo()
    model_.par.max_iter_broyden = 40
    model_.find_transition_path(do_print=True)
    print('')
    dQ_init = model_.path.Q[0,0]/model_.ss.Q-1
    model_.par.scale = 0.01 / dQ_init    
    

paths = ['Y','C','C_T', 'C_NT', 'wnT', 'wnNT', 'YT', 'YNT']
pathlabels = ['Output', 'Agg. C', 'Avg. C in tradeable sec.', 'Avg. C in Non-tradeable sec.',
              'Real income T', 'Real income NT', 'Tradeable Output', 'Non-tradeable Output']

colors=['Blue', 'firebrick', 'darkgreen']
labels = [r'No endo. uncertainty', f'$\epsilon_y=0.1$', f'$\epsilon_y=1$']

fig = figs.show_IRFs_new(models=models, paths=paths,labels=labels, pathlabels=pathlabels, colors=colors,
                    T_max=40, do_sumplot=False, scale=True, ldash=['-', '--', '-.'], lwidth=2)  

fig.savefig('plots\Endo_uncertainty.pdf')

sige = model_HA_no_uncertainty.par.sigma_e
plt.plot((model_HA_no_uncertainty.par.sigma_e_trans[:40,0]/sige-1)*100, color='blue', label=labels[0])
plt.plot((model_HA_no_uncertainty.par.sigma_e_trans[:40,1]/sige-1)*100, '--', color='blue', label=labels[0])
plt.plot((model_HA_uncerntainty_T_NT.par.sigma_e_trans[:40,0]/sige-1)*100, color='orange', label=labels[1])
plt.plot((model_HA_uncerntainty_T_NT.par.sigma_e_trans[:40,1]/sige-1)*100, '--', color='orange', label=labels[1])
plt.plot((model_HA_uncerntainty_T_NT_large.par.sigma_e_trans[:40,0]/sige-1)*100, color='red', label=labels[2])
plt.plot((model_HA_uncerntainty_T_NT_large.par.sigma_e_trans[:40,1]/sige-1)*100, '--', color='red', label=labels[2])
#plt.ylim(0.4, 0.8)
plt.legend()
plt.tight_layout()
plt.show()



del model_HA_no_uncertainty, model_HA_uncerntainty_T_NT, model_HA_uncerntainty_T_NT_large



