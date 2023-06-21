# -*- coding: utf-8 -*-
import os
os.environ["NUMBA_PARFOR_MAX_TUPLE_SIZE"] = "200"
import sys
sys.path.append("..")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,'..')

import time
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt   
from IHANKModel import HANKModelClass
import utils
import figs
import GetForeignEcon
import numba as nb
nb.set_num_threads(4)


#%%

model = HANKModelClass(name='baseline')

# HH model type
# model.par.HH_type = 'RA-CM'
model.par.HH_type = 'RA-IM'
# model.par.HH_type = 'TA-CM'
# model.par.HH_type = 'TA-IM'
# model.par.HH_type = 'HA'


model.find_ss(do_print=True)
assert abs(model.ss.Walras) < 1e-07
if model.par.HH_type == 'HA':
    utils.print_MPCs(model, do_print=True)


save_initvals = False
if save_initvals:
    np.savez('saved/Va_init.npz', Va_init=model.ss.Va)


# figs.PE_MonPol_shock(model, lin=False)
# figs.plot_MPCs(model, lin=False)
# figs.PE_MonPol_shock_Holm(model, lin=False)


#%%

#model.par.floating = False   
#model.par.r_debt_elasticity = 0.
#model.par.use_RA_jac = False
#model.par.VAT_weight = 0.5
if model.par.HH_type == 'HA':
    model.compute_jacs(do_print=True,skip_shocks=True,skip_hh=False)
else:
    model.compute_jacs(do_print=True,skip_shocks=True,skip_hh=True)



#%%

utils.Create_shock('FD', model, 0.001, absval=True)
model.path.FD[0,:] = 0.
model.path.FD[0,:] = 0.01
model.transition_path(do_print=True)
utils.scaleshock('C', model,0.01)


#%%
model_nomdeval = model.copy()

#model_nomdeval.path.ND[0,:] =  0.001*rho**np.arange(model_nomdeval.par.T)
utils.Create_shock('ND', model_nomdeval, 0.001, absval=True)
model_nomdeval.path.ND[0,:] = 1.
model_nomdeval.path.ND[0,:] = 1 + 0.01
model_nomdeval.transition_path(do_print=True)
utils.scaleshock('C', model_nomdeval,0.01)

#%%

abs_value = ['iF_s', 'goods_mkt', 'NKPC', 'Taylor_rule', 'StockPrice', 'E', 'NFA', 'Dompi', 'iF_s', 'pi', 'piH', 'piF', 'UniformT', 'NFA', 'NX', 'VAT', 'piF_s', 'rF_s', 'r']
#paths = ['Y','C','r', 'ra', 'Walras', 'A',  'Q', 'pi', 'piNT', 'piH', 'wnT', 'wnNT', 'YT', 'YNT', 'NT', 'NNT', 'B', 'LT', 'G', 'YT', 'G_T', 'G_NT', 'NX', 'PF_s', 'C_s', 'iF_s'] 
paths = ['VA','VA_NT','VA_T','C', 'N', 'r', 'E',  'wnNT', 'NT',  'CT',  'ToT', 'Q'] 
#paths = ['VA','VA_NT','VA_T','C', 'N', 'r', 'E',  'wnNT', 'NT',  'CT',  'ToT', 'Q', 'wnT', 'wnNT', 'INC_T_hh', 'INC_NT_hh'] 
figs.show_IRFs_new([model, model_nomdeval], paths,abs_value=abs_value,T_max=30, labels=['FD', 'ND'], shocktitle='Devaluations', do_sumplot=False, scale=True, ldash=['-', '--'])


#%%

model.par.floating = True   
model.par.TaylorType = 'FoF'
model.par.use_RA_jac = True
model.par.r_debt_elasticity = 0.
model.par.phi_back = 1e-05
if model.par.HH_type == 'HA':
    model.compute_jacs(do_print=True,skip_shocks=True,skip_hh=False)
else:
    model.compute_jacs(do_print=True,skip_shocks=True,skip_hh=True)

model_foreign = GetForeignEcon.get_foreign_econ(shocksize=-0.01)
GetForeignEcon.create_foreign_shock(model, model_foreign)


model.transition_path(do_print=True)
utils.scaleshock('C_s', model,0.01)


abs_value = ['iF_s', 'goods_mkt', 'NKPC', 'Taylor_rule', 'StockPrice', 'E', 'NFA', 'Dompi', 'iF_s', 'pi', 'piH', 'piF', 'UniformT', 'NFA', 'NX', 'VAT', 'piF_s', 'rF_s', 'r']
# paths = ['C_s','iF_s','piF_s', 'PF_s']  
#figs.show_IRFs_new([model], paths,abs_value=abs_value,T_max=40, shocktitle='Demand shock', scale=True, do_sumplot=False)
#paths = ['Y','C','r', 'ra', 'Walras', 'A',  'Q', 'pi', 'piNT', 'piH', 'wnT', 'wnNT', 'YT', 'YNT', 'NT', 'NNT', 'B', 'LT', 'G', 'YT', 'G_T', 'G_NT', 'NX', 'PF_s', 'C_s', 'iF_s'] 
paths = ['VA','VA_NT','VA_T','C', 'r', 'Walras',  'wnNT', 'NT',  'CT', 'CNT', 'NFA', 'B', 'XM2T', 'CF', 'Q', 'VAT', 'subP', 'Exports','Imports', 'ToT', 'E' ] 
#abels =  ['VAT','VANT','GDP','C', 'r', 'Walras',  'wnNT', 'wnT',   'CT', 'CNT', 'NFA', 'B', 'rFs', 'Cs', 'Q', 'VAT', 'E', 'CHs', 'logCNThome', 'logCThome'] 
figs.show_IRFs_new([model], paths,abs_value=abs_value,T_max=30, shocktitle='Demand shock', do_sumplot=False, scale=True)



#%%

#model.par.floating = False   
#model.compute_jacs(do_print=True,skip_shocks=True,skip_hh=False)

model_nomdeval = model.copy()

utils.Create_shock('ND', model_nomdeval, 0.001, absval=True)
model_nomdeval.transition_path(do_print=True)
utils.scaleshock('ToT', model_nomdeval,0.01)

utils.Create_shock('deval', model, 0.001, absval=True)
model.transition_path(do_print=True)
utils.scaleshock('ToT', model,0.01)

#%%
abs_value = ['iF_s', 'goods_mkt', 'NKPC', 'Taylor_rule', 'StockPrice', 'E', 'NFA', 'Dompi', 'iF_s', 'pi', 'piH', 'piF', 'UniformT', 'NFA', 'NX', 'VAT', 'piF_s', 'rF_s', 'r']
#paths = ['Y','C','r', 'ra', 'Walras', 'A',  'Q', 'pi', 'piNT', 'piH', 'wnT', 'wnNT', 'YT', 'YNT', 'NT', 'NNT', 'B', 'LT', 'G', 'YT', 'G_T', 'G_NT', 'NX', 'PF_s', 'C_s', 'iF_s'] 
paths = ['VA','VA_NT','VA_T','C', 'N', 'r', 'E',  'wnNT', 'NT',  'CT',  'ToT', 'Q', 'wnT', 'wnNT', 'INC_T_hh', 'INC_NT_hh'] 
figs.show_IRFs_new([model, model_nomdeval], paths,abs_value=abs_value,T_max=30, labels=['FD', 'ND'], shocktitle='Devaluations', do_sumplot=False, scale=True, ldash=['-', '--'])


#%%

model.use_FD_shock(False)
model.compute_jacs(do_print=True,skip_shocks=False,skip_hh=False)

#%%
finjac = utils.get_GE_mat(model, ['ToT'], ['VAT', 'subP'], calc_jac=False)
#dToT_lin = finjac[('ToT', 'deval')] @ (model.path.deval[0,:] - model.ss.deval)

#%%
dToT = model_nomdeval.path.ToT[0,:] - model.ss.ToT 
dToT_dVAT = finjac[('ToT', 'VAT')] @ 0.8**np.arange(model.par.T)
dToT_dsubP = finjac[('ToT', 'subP')] @ 0.8**np.arange(model.par.T)
Nquarter = 10

#%%
from scipy.optimize import minimize

def ToT_res(x):
    scale, weight = x
    
    implied_ToT = scale * (weight*dToT_dVAT + (1-weight)*dToT_dsubP)
    
    res = 10*np.sum(abs(implied_ToT[:Nquarter]-dToT[:Nquarter]))
    return res 

x0 = np.array([0.01,0.5]) 
result = minimize(ToT_res, x0, method='Nelder-Mead', tol=1e-6)
print(result)

scale, weight = result.x
implied_ToT = scale * (weight*dToT_dVAT + (1-weight)*dToT_dsubP)
plt.plot(dToT[:30])
plt.plot(implied_ToT[:30], '--')
plt.show()


#%%

plt.plot((model.path.ToT[0,:40]-model.ss.ToT)*100)
plt.plot(dToT_lin[:40]*100, '--')
plt.show()

#%%


model.par.floating = True   
if model.par.HH_type == 'HA':
    model.compute_jacs(do_print=True,skip_shocks=True,skip_hh=False)
else:
    model.compute_jacs(do_print=True,skip_shocks=True,skip_hh=True)

model_foreign = GetForeignEcon.get_foreign_econ(shocksize=-0.01)
GetForeignEcon.create_foreign_shock(model, model_foreign)


model.transition_path(do_print=True)
utils.scaleshock('C_s', model,0.01)

#%%
from copy import deepcopy

#rho_list = [0.5, 0.6, 0.7, 0.8, 0.9] # , 0.95
rho_list = [0.5, 0.6, 0.7, 0.8] # , 0.95
irfs ={} 
for rho in rho_list:
    print(rho)
    model_foreign = GetForeignEcon.get_foreign_econ(shocksize=-0.01, persistence=rho)
    GetForeignEcon.create_foreign_shock(model, model_foreign)   
    model.transition_path(do_print=True)
    irfs[str(rho)] = deepcopy(model.path)
    
#%%
    
var = 'C'
t_plot = 150
irf_out = np.zeros((5,t_plot)) 
for i,rho in enumerate(rho_list):
    irf_out[i,:] = 100*(getattr(irfs[str(rho)],var)/getattr(model.ss,var)-1)[0,:t_plot]
    
plt.plot(irf_out.T)
plt.show()

#%%


rho = 0.9 

#%%

model_tshort =  HANKModelClass(name='baseline', par={'T' : 300})
model_tshort.par.HH_type = 'HA'
model_tshort.find_ss(do_print=False)    
model_tshort.compute_jacs(do_print=True,skip_shocks=False,skip_hh=False)

#%%
model_foreign = GetForeignEcon.get_foreign_econ(shocksize=-0.01, upars={'T' : 300}, persistence=rho)
GetForeignEcon.create_foreign_shock(model_tshort, model_foreign)
model_tshort.transition_path(do_print=False)
utils.Create_shock('C_s', model_tshort, 0.001)

#%%

#model_tshort.prepare_simulate(skip_hh=True,only_pols_hh=True,reuse_G_U=False,do_print=True)

totvarlist = ['C', 'Y', 'Q']
shocks_exo = ['C_s','piF_s', 'iF_s']
GE_jac = utils.get_GE_mat(model_tshort, totvarlist, shocks_exo, calc_jac=False)

#%%

shock = 'piF_s'
T = 300
rhos = np.array([0.2, 0.4, 0.6, 0.8, 0.9, 0.95])
dshock = 0.005 * rhos ** (np.arange(T)[:, np.newaxis])
dC_lin = GE_jac[('C', shock)] @ dshock

dC_nonlin = np.zeros_like(dC_lin)

for k in range(len(rhos)):
    custompath = {shock : dshock[:,k]}
    utils.Create_shock(shock, model_tshort, custompath=custompath)
    model_tshort.transition_path(do_print=False)
    dC_nonlin[:,k] = model_tshort.path.C[0,:] - model_tshort.ss.C

#%%
plt.plot(100*dC_lin[:75, :])
plt.plot(100*dC_nonlin[:75, :], '--')
plt.show()


•#%%

model_tlong =  HANKModelClass(name='baseline', par={'T' : 900})
model_tlong.par.HH_type = 'HA'
model_tlong.find_ss(do_print=False)    
model_tlong.compute_jacs(do_print=True,skip_shocks=True,skip_hh=False)
#%%
model_foreign = GetForeignEcon.get_foreign_econ(shocksize=-0.01, upars={'T' : 900}, persistence=rho)
GetForeignEcon.create_foreign_shock(model_tlong, model_foreign)
model_tlong.transition_path(do_print=True)
utils.Create_shock('C_s', model_tlong, 0.001)


#%%

abs_value = ['iF_s', 'goods_mkt', 'NKPC', 'Taylor_rule', 'StockPrice', 'E', 'NFA', 'Dompi', 'iF_s', 'pi', 'piH', 'piF', 'UniformT', 'NFA', 'NX', 'VAT', 'piF_s', 'rF_s', 'r']
paths = ['Y','C','C_s' ] 
figs.show_IRFs_new([model_tshort, model_tlong], paths,abs_value=abs_value,T_max=30, labels=['short', 'long'], shocktitle='Devaluations', do_sumplot=False, scale=False, ldash=['-', '--'])

    
#%%

utils.Create_shock('deval', model, 0.001, absval=True)
model.path.deval[0,:] = 0.
model.path.deval[0,:] = 0.01
model.transition_path(do_print=True)
utils.scaleshock('C', model,0.01)


#model_nomdeval.path.ND[0,:] =  0.001*rho**np.arange(model_nomdeval.par.T)
utils.Create_shock('ND', model_nomdeval, 0.001, absval=True)
model_nomdeval.path.ND[0,:] = 1.
model_nomdeval.path.ND[0,:] = 1 + 0.01
model_nomdeval.transition_path(do_print=True)
utils.scaleshock('C', model_nomdeval,0.01)


#%%



#%%

abs_value = ['iF_s', 'goods_mkt', 'NKPC', 'Taylor_rule', 'StockPrice', 'E', 'NFA', 'Dompi', 'iF_s', 'pi', 'piH', 'piF', 'UniformT', 'NFA', 'NX', 'VAT', 'piF_s', 'rF_s', 'r']
#paths = ['Y','C','r', 'ra', 'Walras', 'A',  'Q', 'pi', 'piNT', 'piH', 'wnT', 'wnNT', 'YT', 'YNT', 'NT', 'NNT', 'B', 'LT', 'G', 'YT', 'G_T', 'G_NT', 'NX', 'PF_s', 'C_s', 'iF_s'] 
paths = ['VA','VA_NT','VA_T','C', 'N', 'r', 'E',  'wnNT', 'NT',  'CT',  'ToT', 'Q'] 
figs.show_IRFs_new([model, model_nomdeval], paths,abs_value=abs_value,T_max=30, labels=['FD', 'ND'], shocktitle='Devaluations', do_sumplot=False, scale=True, ldash=['-', '--'])


#%%

figs.PE_MonPol_shock(model, lin=False)
figs.plot_MPCs(model, lin=False)

#%%

model._compute_jac_hh(do_print=True)


#%%
rho = 0.8
shock = 0.01*rho**np.arange(model.par.T)


dCT_dLI= model.jac_hh[('C_T_hh','wnT')] #@ shock 
dCNT_dLI  = model.jac_hh[('C_NT_hh','wnNT')] #@ shock 


ss= model.ss
par=model.par 

# linearie labor income:
dI_dNT = ss.wT  / par.sT * shock
dI_dNNT = ss.wNT / (1-par.sT ) * shock

dI_dwT = ss.NT  / par.sT * shock
dI_dwNT = ss.NNT  /  (1-par.sT ) * shock

dC_dN = dCT_dLI @ dI_dNT + dCNT_dLI @ dI_dNNT
dC_dw = dCT_dLI @ dI_dwT + dCNT_dLI @ dI_dwNT

tplot=15
plt.plot(dC_dN[:tplot]/ss.C*100)
plt.plot(dC_dw[:tplot]/ss.C*100, '--')
plt.show()



#%%


model.solve_hh_ss()
model.simulate_hh_ss()

C_ss = np.sum(model.ss.C*model.ss.D)
print(C_ss)
A = np.sum(model.ss.A*model.ss.D)
print(A)



fig, ax = plt.subplots()
ax.hist(model.ss.a.flatten(), 100, weights=model.ss.D.flatten(), range=(amin,2))
plt.show()


#%%

figs.plot_MPCs(model, lin=False)
#%%
model_HA = model.copy()
lsize = 1.5
Nquarters = 30 

HA_dC_dI = model_HA.jac_hh[('C_hh', 'UniformT')]
HA_dC_dr = model_HA.jac_hh[('C_hh', 'ra')]

RA_dC_dI = model_RA.par.M_Y.copy() 
RA_dC_dr = model_RA.par.M_R.copy() 
# Convert RA_dC_dr to ra dating 
RA_dC_dr[:,1:] = RA_dC_dr[:,:-1].copy()
RA_dC_dr[:,0] = RA_dC_dI[:,0] * model_RA.ss.r

columns = [0,4, 8, 12, 16, 20]
alphalist = np.flip(np.linspace(0.4,1,len(columns)))
l1,l2 = 'HANK', 'RANK'

fig = plt.figure(figsize=(8,3.0))

ax = fig.add_subplot(1,2,1)
# plt.plot(x1, y1, 'ko-')
ax.set_title(r'$\mathbf{M}$')

l1_c,l2_c = l1, l2
for col in columns:   
    ax.plot(HA_dC_dI[:Nquarters, col], '-', label=l1_c, color='C2', linewidth=lsize)
    ax.plot(RA_dC_dI[:Nquarters, col], '-', label=l2_c, color='C1', linewidth=lsize)  

ax.plot(np.zeros(Nquarters), '--', color='black')

ax.set_xlabel('Quarters', fontsize=12)
# plt.ylabel('Damped oscillation')
#ax.legend(frameon=True, fontsize=10)
ax.set_ylabel(f'$dC$', fontsize=12)


ax = fig.add_subplot(1,2,2)
ax.set_title(r'$\mathbf{M_r}$')

l1_c,l2_c = l1, l2
for i,col in enumerate(columns):   
    ax.plot(HA_dC_dr[:Nquarters, col], '-', label=l1_c, color='C2', linewidth=lsize, alpha=alphalist[i])
    ax.plot(RA_dC_dr[:Nquarters, col], '-', label=l2_c, color='C1', linewidth=lsize, alpha=alphalist[i])    
    l1_c, l2_c = '_nolegend_', '_nolegend_' # suppress extra legends

ax.plot(np.zeros(Nquarters), '--', color='black')

ax.set_xlabel('Quarters', fontsize=12)
# plt.ylabel('Damped oscillation')
ax.legend(frameon=True, fontsize=12)
ax.set_ylabel(f'$dC$', fontsize=12)


plt.tight_layout()

plt.show()

#%%

par=model.par
ss = model.ss 
amax=100

for i_beta,beta in enumerate(par.beta_grid):
    
    fig = plt.figure(figsize=(12,4),dpi=100)

    I = par.a_grid < 2

    # a. consumption
    ax = fig.add_subplot(1,2,1)
    ax.set_title(f'consumption ($\\beta = {beta:.4f}$)')

    for i_z,z in enumerate(par.z_grid_ss):
        if i_z%3 == 0 or i_z == par.Nz-1:
            ax.plot(par.a_grid[I],ss.c[i_beta,i_z,I],label=f'z = {z:.2f}')

    ax.legend(frameon=True)
    ax.set_xlabel('savings, $a_{t-1}$')
    ax.set_ylabel('consumption, $c_t$')

    # b. saving
    ax = fig.add_subplot(1,2,2)
    ax.set_title(f'saving ($\\beta = {beta:.4f}$)')

    for i_z,z in enumerate(par.z_grid_ss):
        if i_z%3 == 0 or i_z == par.Nz-1:
            ax.plot(par.a_grid[I],ss.a[i_beta,i_z,I]-par.a_grid[I],label=f'z = {z:.2f}')

    ax.set_xlabel('savings, $a_{t-1}$')
    ax.set_ylabel('savings change, $a_{t}-a_{t-1}$')

    fig.tight_layout()
    
#%%
a_grid = model.par.a_grid


plt.plot(a_grid)
plt.plot()

#%%

def a_grid_borrow_prem(grid):
    amin = grid.flat[np.abs(grid - 0).argmin()]
    if amin>0:
        aneg = np.abs(grid - 0).argmin() -1
        apos = np.abs(grid - 0).argmin()
    else:
        aneg = np.abs(grid - 0).argmin()
        apos = np.abs(grid - 0).argmin() + 1  
    
    
    grid[aneg] = -1e-05
    grid[apos] = 1e-05
    return grid    

Na = model.par.Na
amax = model.par.a_max
amin = model.par.a_min

a_grid_test = np.zeros(Na)

def nonlinspace(amax, n, phi, amin=0):
    """Create grid between amin and amax. phi=1 is equidistant, phi>1 dense near amin. Extra flexibility may be useful in non-convex problems in which policy functions have nonlinear (even non-monotonic) sections far from the borrowing limit."""
    a_grid = np.zeros(n)
    a_grid[0] = amin
    for i in range(1, n):
        a_grid[i] = a_grid[i-1] + (amax - a_grid[i-1]) / (n-i)**phi 
    return a_grid


N_lin = 100
N_lin_break = 3
a_grid_test[:-N_lin] = np.linspace(0,N_lin_break,Na-N_lin)
a_grid_test[Na-N_lin:] = nonlinspace(amax, N_lin, 1.6, amin=N_lin_break)
a_grid_test = a_grid_borrow_prem(a_grid_test)

plt.plot(a_grid)
plt.plot(a_grid_test)
plt.plot()

#%%
from copy import deepcopy 

model._set_inputs_hh_all_ss()
model.path.wnT[:,:] = deepcopy(model_TANK.path.wnT[:,:])
model.path.wnNT[:,:] = deepcopy(model_TANK.path.wnNT[:,:])
model.path.ra[:,:] = deepcopy(model_TANK.path.ra[:,:])
model.path.UniformT[:,:] = deepcopy(model_TANK.path.UniformT[:,:])
model.path.LT[:,:] = deepcopy(model_TANK.path.LT[:,:])

model.solve_hh_path()
model.simulate_hh_path()

#%%

dC = np.zeros(model.par.T)
for t in range(model.par.T):
    dC[t] = np.sum(model.path.D[t,...]*model.path.c[t,...]) - model.ss.C
    
plt.plot(dC[:30]/model.ss.C*100)    
plt.show() 


#%%

model._set_inputs_hh_all_ss()
model.path.UniformT[:,:] = 0.0 
model.path.UniformT[0,0] = 0.05 * (model.ss.WT * model.ss.NT + model.ss.WNT * model.ss.NNT) #  500$
dI = model.path.UniformT[0,0]      
model.solve_hh_path()
model.simulate_hh_path()


MPC = (model.path.c[0,:,:,:]-model.ss.c)/model.path.UniformT[0,0]


agrid = model.par.a_grid
I = agrid<2
plt.plot(agrid[I], MPC[0,0,I])
plt.plot(agrid[I],MPC[1,2,I], '--')
plt.plot(agrid[I],MPC[2,0,I],'-.')
plt.plot(agrid[I],MPC[3,3,I])
plt.show()


#%%



#model.par.phi_back = 0.7
if model.par.HH_type == 'HA':
    model.compute_jacs(do_print=True,skip_shocks=True,skip_hh=False)
else:
    model.compute_jacs(do_print=True,skip_shocks=True,skip_hh=True)


#%%

upars = {'use_matched_model' : False}
model.par.beta_corr = 0.0 
model_foreign = GetForeignEcon.get_foreign_econ(shocksize=-0.01, upars=upars)
GetForeignEcon.create_foreign_shock(model, model_foreign)


model.transition_path(do_print=True)
utils.scaleshock('C_s', model,0.01)

abs_value = ['iF_s', 'goods_mkt', 'NKPC', 'Taylor_rule', 'StockPrice', 'E', 'NFA', 'Dompi', 'iF_s', 'pi', 'piH', 'piF', 'UniformT', 'NFA', 'NX', 'VAT', 'piF_s', 'rF_s', 'r']
# paths = ['C_s','iF_s','piF_s', 'PF_s']  
#figs.show_IRFs_new([model], paths,abs_value=abs_value,T_max=40, shocktitle='Demand shock', scale=True, do_sumplot=False)
#paths = ['Y','C','r', 'ra', 'Walras', 'A',  'Q', 'pi', 'piNT', 'piH', 'wnT', 'wnNT', 'YT', 'YNT', 'NT', 'NNT', 'B', 'LT', 'G', 'YT', 'G_T', 'G_NT', 'NX', 'PF_s', 'C_s', 'iF_s'] 
paths = ['VA','VA_NT','VA_T','C', 'r', 'Walras',  'wnNT', 'NT',  'CT', 'CNT', 'NFA', 'B', 'XM2T', 'CF', 'Q', 'VAT', 'subP', 'Exports','Imports', 'ToT' ] 
#abels =  ['VAT','VANT','GDP','C', 'r', 'Walras',  'wnNT', 'wnT',   'CT', 'CNT', 'NFA', 'B', 'rFs', 'Cs', 'Q', 'VAT', 'E', 'CHs', 'logCNThome', 'logCThome'] 
figs.show_IRFs_new([model], paths,abs_value=abs_value,T_max=30, shocktitle='Demand shock', do_sumplot=False, scale=True)


#%%

model_TANK = model.copy()

#%%


model_RA = model.copy()
model_RA.par.HH_type = 'RA-IM'
fig = figs.C_decomp_HA_v_RA([model,model_RA], T_max=30, lwidth=2.5, testplot=False, disp_income=True)

#%%

model._set_inputs_hh_all_ss()
model.path.UniformT[:,:] = 0.0 
model.path.UniformT[0,0] = 0.05 * (model.ss.WT * model.ss.NT + model.ss.WNT * model.ss.NNT) #  500$
dI = model.path.UniformT[0,0]      
model.solve_hh_path()
model.simulate_hh_path()

#%%

MPC = (model.path.c[0,:,:,:]-model.ss.c)/model.path.UniformT[0,0]
amax=100
agrid = model.par.a_grid
plt.plot(agrid[:amax], MPC[0,0,:amax])
plt.plot(agrid[:amax],MPC[1,0,:amax])
plt.plot(agrid[:amax],MPC[2,0,:amax])
plt.plot(agrid[:amax],MPC[4,0,:amax])
plt.show()


#%%

model_float = model.copy()
model_float.par.floating = True
if model.par.HH_type == 'HA':
    model_float.compute_jacs(do_print=True,skip_shocks=True,skip_hh=False)
else:
    model_float.compute_jacs(do_print=True,skip_shocks=True,skip_hh=True)
    
model_float.par.beta_corr = 0.0 
GetForeignEcon.create_foreign_shock(model_float, model_foreign)

model_float.transition_path(do_print=True)
utils.scaleshock('C_s', model_float, 0.01)    

    
#%%

model.ss.rel = model.ss.PF/model.ss.PH
model_float.ss.rel = model_float.ss.PF/model_float.ss.PH

model.path.rel = np.zeros_like(model.path.N)
model_float.path.rel = np.zeros_like(model.path.N)

model.path.rel =  model.path.PF/model.path.PH 
model_float.path.rel =  model_float.path.PF/model_float.path.PH 

model.IRF['rel'] =  model.path.PF/model.path.PH 
model_float.IRF['rel'] =  model_float.path.PF/model_float.path.PH 

labels = ['Fixed', 'float']
varnames = ['VA_T', 'VA_NT', 'C', 'Q', 'rel', 'ToT', 'PF', 'PH']
model.compare_IRFs(models=[model, model_float],labels=labels,varnames=varnames,T_max=40, do_shocks=False,do_targets=False)



#%%


delta = dE = 0.05 * 0.8** np.arange(model.par.T)
AR_path = 0.8** np.arange(model.par.T)

ss  = model.ss
VAT_revenue =  (ss.PNT * ss.VA_NT + ss.PH * ss.VA_T) / ss.P 
sub_cost =  (ss.WT*ss.NT + ss.WNT*ss.NNT) / ss.P


VAT_level = 0.03
VAT  = AR_path * VAT_level
subP = AR_path * VAT_level/sub_cost

#custompath = {'VAT' : VAT }
#utils.Create_shock(['VAT'], model, 0.001, absval=False, custompath=custompath)

custompath = {'VAT' : VAT, 'subP' : subP}
utils.Create_shock(['VAT', 'subP'], model, custompath=custompath)

#%% delta deval 

delta = 0.03 * 0.8** np.arange(model.par.T)
VAT = delta/(1+delta)
subP = delta/(1+delta)

custompath = {'VAT' : VAT, 'subP' : subP}
utils.Create_shock(['VAT', 'subP'], model, custompath=custompath)

#%%

deval = 0.03 * 0.8** np.arange(model.par.T)

custompath = {'deval' : deval}
utils.Create_shock(['deval'], model, custompath=custompath)

model.transition_path(do_print=True)


#%%

abs_value = ['iF_s', 'goods_mkt', 'NKPC', 'Taylor_rule', 'StockPrice', 'E', 'NFA', 'Dompi', 'iF_s', 'pi', 'piH', 'piF', 'UniformT', 'NFA', 'NX', 'VAT', 'piF_s', 'rF_s', 'r']
# paths = ['C_s','iF_s','piF_s', 'PF_s']  
#figs.show_IRFs_new([model], paths,abs_value=abs_value,T_max=40, shocktitle='Demand shock', scale=True, do_sumplot=False)
#paths = ['Y','C','r', 'ra', 'Walras', 'A',  'Q', 'pi', 'piNT', 'piH', 'wnT', 'wnNT', 'YT', 'YNT', 'NT', 'NNT', 'B', 'LT', 'G', 'YT', 'G_T', 'G_NT', 'NX', 'PF_s', 'C_s', 'iF_s'] 
paths = ['VA','VA_NT','VA_T','C', 'r', 'Walras',  'wnNT', 'NT',  'CT', 'CNT', 'NFA', 'B', 'XM2T', 'CF', 'Q', 'VAT', 'subP', 'Exports','Imports', 'ToT' ] 
#abels =  ['VAT','VANT','GDP','C', 'r', 'Walras',  'wnNT', 'wnT',   'CT', 'CNT', 'NFA', 'B', 'rFs', 'Cs', 'Q', 'VAT', 'E', 'CHs', 'logCNThome', 'logCThome'] 
figs.show_IRFs_new([model], paths,abs_value=abs_value,T_max=30, shocktitle='Demand shock', do_sumplot=False, scale=False)



#%%

path=model.path
ss=model.ss
ToT = path.PF[0,:]/path.PH_s[0,:] / path.E[0,:] * (1-path.VAT[0,:])
ToTss = ss.PF/ss.PH_s / ss.E * (1-ss.VAT)

plt.plot((ToT[:30]/ToTss-1)*100)
plt.show()

#%%
dInc = model.path.Income[0,:]-model.ss.Income

dC_dY = model.par.M_Y @ dInc

dr = model.path.r[0,:]-model.ss.r
dC_dr = model.par.M_R @ dr

dC = model.path.C[0,:]-model.ss.C
plt.plot(dC[:20])
plt.plot(dC_dY[:20])
plt.plot(dC_dr[:20])
plt.show()

#%%

from scipy.interpolate import CubicSpline

MPC_data = {}
MPC_data['x'] = np.arange(6)
MPC_data['y'] =  np.array([0.554, 0.195, 0.124, 0.085, 0.064, 0.057])

cs = CubicSpline(MPC_data['x'] , MPC_data['y'] )
Nquarters = 11 

MPC_data['x_int'] = np.arange(Nquarters)/4
MPC_data['x_int_Q'] = np.arange(Nquarters)

y_int = cs(MPC_data['x_int'])/4
MPC_data['y_int'] = y_int / np.sum(y_int[:4]) * MPC_data['y'][0] 


plt.plot(MPC_data['x_int'], MPC_data['y_int'])
plt.show()


#%%
### IMPCS


# array([0.10327966, 0.9798307 , 0.00658124])

beta_delta = 0.00658124
model.par.sLiquid = 0.10327966
model.par.beta_mean, model.par.beta_delta =  np.array([ 0.96654, beta_delta])

model.solve_hh_ss(do_print=False)
model.simulate_hh_ss(do_print=False)  

ss=model.ss
ss.A = np.sum(ss.D*ss.a)
ss.C = np.sum(ss.D*ss.c)
print(ss.A/model.ss.Income, ss.C)
figs.plot_MPCs(model, lin=False)


#%%

from scipy import optimize
from scipy.optimize import minimize
from scipy.optimize import least_squares
from scipy.optimize import LinearConstraint

#beta_delta = 0.02

def HH_obj(x):
    print(x)
    beta_delta = x[2]
    if x[1] + beta_delta > 0.99346249:
        x[1] = 0.99346249-beta_delta
        pen = 50
    else:
        pen=0
    
    model.par.sLiquid = x[0]
    model.par.beta_mean, model.par.beta_delta =  np.array([x[1], beta_delta])
    
    model.solve_hh_ss(do_print=False)
    model.simulate_hh_ss(do_print=False)  
    
    ss=model.ss
    ss.A = np.sum(ss.D*ss.a)
    ss.C = np.sum(ss.D*ss.c)
    #print(ss.A/model.ss.Income, ss.C)
    #figs.plot_MPCs(model, lin=False)    

    MPC_ann = utils.nonlin_MPC(model)[0]
    MPC_res1 = (MPC_ann[0] - model.par.Agg_MPC)*30
    MPC_res2 = (MPC_ann[1] - 0.2)*40
        
    #return np.array([MPC_res1, MPC_res2, ss.A/model.ss.Income-12])
    return sum(abs(np.array([MPC_res1, MPC_res2, ss.A/model.ss.Income-12]))) + pen

def HH_obj_OLS(x):
    print(x)
    beta_delta = 0.00658124

    try:
        model.par.sLiquid = x[0]
        model.par.beta_mean, model.par.beta_delta =  np.array([x[1], beta_delta])
        
        model.solve_hh_ss(do_print=False)
        model.simulate_hh_ss(do_print=False)  
        pen=0
    except:
        x[1] = 0.96654
        model.par.sLiquid = x[0]
        model.par.beta_mean, model.par.beta_delta =  np.array([x[1], beta_delta])
        
        model.solve_hh_ss(do_print=False)
        model.simulate_hh_ss(do_print=False) 
        pen = 50         
        
    
    ss=model.ss
    ss.A = np.sum(ss.D*ss.a)
    ss.C = np.sum(ss.D*ss.c)
    #print(ss.A/model.ss.Income, ss.C)
    #figs.plot_MPCs(model, lin=False)    

    MPC_ann = utils.nonlin_MPC(model)[0]
    MPC_res1 = (MPC_ann[0] - model.par.Agg_MPC)*30
    MPC_res2 = (MPC_ann[1] - 0.2)*40
        
    return np.array([MPC_res1, MPC_res2, ss.A/model.ss.Income-12])
    #return sum(abs(np.array([MPC_res1, MPC_res2, ss.A/model.ss.Income-12]))) + pen


#%%
# HH_obj([0.25, 0.986])


x0 = [0.10327966, 0.9668307]
#sol = optimize.root(HH_obj, [0.25, 0.986, 0.00048],  method='hybr')
#sol = minimize(HH_obj, [0.25, 0.986, 0.00048],  method='L-BFGS-B')
sol = least_squares(HH_obj_OLS, x0, bounds=([0.02, 0.7], [0.99, 0.994-beta_delta]), loss='linear') 
print(sol)


#%%
from scipy.optimize import NonlinearConstraint
def constr_f(x):
    return np.array(x[1] + x[2])

#cons = [{'type': 'ineq', 'fun': lambda x:  x[1] + x[2] - 0.99346249}]
nlc = NonlinearConstraint(constr_f, 0.1, 0.99346249)

from scipy.optimize import rosen, differential_evolution
bounds = [(0.02,0.99), (0.7, 0.999-0.01), (0.0, 0.04)]

result = differential_evolution(HH_obj, bounds,  constraints=(nlc), seed=1)
print(result)
result.x, result.fun


#%%

ss = model.ss
par = model.par 

c = ss.c.reshape((par.Nbeta,2,par.Ne,par.Na))
a = ss.a.reshape((par.Nbeta,2,par.Ne,par.Na))

for i_beta,beta in enumerate(par.beta_grid):
    
    fig = plt.figure(figsize=(12,4),dpi=100)

    # a. consumption
    I = (par.a_grid < 40) & (par.a_grid > 15)
    
    ax = fig.add_subplot(1,2,1)
    ax.set_title(f'consumption ($\\beta = {beta:.4f}$)')

    for i_e in range(par.Ne-1):
        ax.plot(par.a_grid[I],c[i_beta,0,i_e,I])

    ax.legend(frameon=True)
    ax.set_xlabel('savings, $a_{t-1}$')
    ax.set_ylabel('consumption, $c_t$')
    
    # b. saving
    #I = par.a_grid < 500
    
    ax = fig.add_subplot(1,2,2)
    ax.set_title(f'saving ($\\beta = {beta:.4f}$)')

    for i_e in range(par.Ne-1):
        ax.plot(par.a_grid[I],a[i_beta,0,i_e,I]-par.a_grid[I])

    
    fig.tight_layout()

#%%
fig = plt.figure(figsize=(12,4),dpi=100)
D = ss.D
ax = fig.add_subplot(1,2,2)
ax.set_title('savings')
for i_beta in range(par.Nbeta):

    label = fr'$\beta = {par.beta_grid[i_beta]:.4f}$'
    y = np.insert(np.cumsum(np.sum(D[i_beta],axis=0)),0,0.0)
    ax.plot(np.insert(par.a_grid,0,par.a_grid[0]),y/y[-1],label=label)
    
ax.set_xlabel('assets, $a_{t}$')
ax.set_ylabel('CDF')
ax.set_xscale('symlog')
ax.legend(frameon=True);


#%%


# model.par.MonPol = 'conR'
# model.par.phi_back = 0.85
# model.par.epsB = 0.5
# model.par.r_debt_elasticity = 0.0
# model.par.r_debt_elasticity = 0.0001
# model.par.tauB = 50
# model.par.TaylorType = 'CPI'
# model.par.phi = 1.0
model.par.phi_back = 0.1

# model.par.debt_rule = True    
# model.par.tax_finance = False 


if model.par.HH_type == 'HA':
    model.compute_jacs(do_print=True,skip_shocks=True,skip_hh=False)
else:
    model.compute_jacs(do_print=True,skip_shocks=True,skip_hh=True)



#%%


model.par.max_iter_broyden = 10
# utils.compute_jac_sym(model, h=1e-04)
# utils.compute_jac_exo_sym(model, h=1e-04)
# utils.Create_shock('UniformT_exo', model, 0.1*model.ss.Y, absval=True)
# utils.Create_shock('betaF', md, -0.0001, absval=False)
# utils.Create_shock('zT', model, 0.001, absval=True)
# utils.Create_shock('G_exo', model, 0.01, absval=True)
# utils.Create_shock('di', model, -0.001, absval=True)
# utils.Create_shock('eps_beta', model, -0.0001, absval=True)
# utils.Create_shock('iF_s', model, -0.01, absval=True)
#utils.Create_shock('C_s', model, 0.05, absval=False)
#utils.Create_shock('VAT', model, -0.05, absval=True)
# utils.Create_shock('subP', model, -0.05, absval=True)


model.par.beta_corr = 0.0 
model_foreign = GetForeignEcon.get_foreign_econ(shocksize=0.01)
GetForeignEcon.create_foreign_shock(model, model_foreign)

# model_foreign = GetForeignEcon.get_foreign_econ(shocksize=0.001, shock='iF_s_exo')
# GetForeignEcon.create_foreign_shock(model, model_foreign)

# model.path.iF_s[:,:] = model.ss.iF_s
# model.path.C_s[:,:] = model.ss.C_s
# model.path.piF_s[:,:] = model.ss.piF_s

# utils.scaleshock('C_s', model,0.01)
# utils.scaleshock('C', model,0.01)

# utils.Create_shock('di', model, 0.001, absval=True)


#%%

delta = dE = 0.05 * 0.8** np.arange(model.par.T)
# delta = dE = 0.01 * 1** np.arange(model.par.T)
AR_path = 0.8** np.arange(model.par.T)

ss  = model.ss

VAT_revenue =  (ss.PNT * ss.VA_NT + ss.PH * ss.VA_T) / ss.P 

sub_cost =  (ss.WT*ss.NT + ss.WNT*ss.NNT) / ss.P




# VAT = delta/(1+delta)
# subP = delta/(1+delta)

# 1 % of GDP 
# VAT  = AR_path * 0.01*ss.GDP/VAT_revenue
# subP = AR_path * 0.01*ss.GDP/sub_cost

VAT_level = 0.03

VAT  = AR_path * VAT_level
subP = AR_path * VAT_level/sub_cost


custompath = {'VAT' : VAT , 'subP' : subP}
utils.Create_shock(['VAT', 'subP'], model, 0.001, absval=False, custompath=custompath)


#%%

# model.path.iF_s[:,:] = model.ss.r 
# model.path.PF_s[:,:] = model.ss.PF_s  
# model.path.C_s[:,:] = model.ss.C_s 
model.par.max_iter_broyden = 15

#model.find_transition_path(shock_specs=shock_specs, do_print=True)

#utils.Create_shock('G_exo', model, 0.01, absval=True)
#model.par.debt_rule = False    

# model.path.di[:,:] = model.ss.di
# model.path.di[0,1] = 0.001
# utils.Create_shock('iF_s', model, 0.0, absval=True)

model.transition_path(do_print=True)
#utils.scaleshock('C', model,0.01)
# comp.find_transition_path(model, do_print=True, use_jac_hh=True)
# utils.scaleshock('Z', model)
# utils.scaleshock('C_s', model,0.01)
# utils.scaleshock('C_s', model)
# utils.scaleshock('iF_s', model,0.01)



#%%
abs_value = ['iF_s', 'goods_mkt', 'NKPC', 'Taylor_rule', 'StockPrice', 'E', 'NFA', 'Dompi', 'iF_s', 'pi', 'piH', 'piF', 'UniformT', 'NFA', 'NX', 'VAT', 'piF_s', 'rF_s', 'r']
# paths = ['C_s','iF_s','piF_s', 'PF_s']  
#figs.show_IRFs_new([model], paths,abs_value=abs_value,T_max=40, shocktitle='Demand shock', scale=True, do_sumplot=False)
#paths = ['Y','C','r', 'ra', 'Walras', 'A',  'Q', 'pi', 'piNT', 'piH', 'wnT', 'wnNT', 'YT', 'YNT', 'NT', 'NNT', 'B', 'LT', 'G', 'YT', 'G_T', 'G_NT', 'NX', 'PF_s', 'C_s', 'iF_s'] 
paths = ['VA_NT','VA_T','GDP','C', 'r', 'Walras',  'wnNT', 'wnT',  'CT', 'CNT', 'NFA', 'B', 'rF_s', 'C_s', 'Q', 'VAT', 'E', 'CH_s', 'CNT_home', 'CT_home'] 
labels =  ['VAT','VANT','GDP','C', 'r', 'Walras',  'wnNT', 'wnT',   'CT', 'CNT', 'NFA', 'B', 'rFs', 'Cs', 'Q', 'VAT', 'E', 'CHs', 'logCNThome', 'logCThome'] 
figs.show_IRFs_new([model], paths,abs_value=abs_value,T_max=30, shocktitle='Demand shock', do_sumplot=False, scale=False, pathlabels=labels)


#%%
plt.plot(delta[:20]*100)
plt.plot((model.path.E[0,:20]-1)*100)
plt.show()


#%%

model.find_IRFs(do_print=True,reuse_G_U=False)

paths = ['C', 'Y']
#model.find_IRFs(do_print=True,reuse_G_U=False,shock_specs={'dPF_s' : model.path.PF_s[0,:]-model.ss.PF_s})
model.show_IRFs(paths,ncols=3,T_max=300,abs_diff=paths,do_linear=True,do_targets=False, do_shocks=False)

#%%
model.par.debt_rule = True   
model.compute_jacs(do_print=True,skip_shocks=False,skip_hh=False)


#%%
# shock_specs={'dPF_s' : model.path.PF_s[0,:]-model.ss.PF_s}
# shock_specs={'dpiF_s' : model.path.piF_s[0,:]-model.ss.piF_s}
shock_specs={'ddi' : model.path.di[0,:]-model.ss.di}
# shock_specs={'dG_exo' : model.path.G_exo[0,:]-model.ss.G_exo}


for var in model.shocks:
    setattr(model.par, 'jump_'+var, 0.0)
paths = ['C', 'Y', 'B']
model.find_IRFs(do_print=True,reuse_G_U=False, shock_specs=shock_specs)
model.show_IRFs(paths,ncols=3,T_max=300,abs_diff=paths,do_linear=True,do_targets=False, do_shocks=False)

# print(model.IRF['PF_s'])


#%%

dPF_s = model.path.PF_s[0,:] / model.path.PH[0,:] -1 #-model.ss.PF_s

plt.plot(dPF_s)
plt.show()


#%%

plt.plot(delta[:20]*100)
plt.plot((model.path.E[0,:20]-1)*100)
plt.show()


#%%

abs_value = [ 'iF_s', 'goods_mkt', 'NKPC', 'Taylor_rule', 'StockPrice', 'E', 'NFA', 'Walras', 'Dompi', 'iF_s', 'pi', 'piH', 'piF', 'UniformT', 'NFA', 'piF_s', 'iF_s', 'iF_s_exo']
paths = ['C_s','iF_s','piF_s', 'PF_s', 'iF_s_exo', 'rF_s']  
labels =  ['Cs','iFs','piFs', 'PFs','iFsexo', 'rFs']  
model_foreign.par.scale = 10
figs.show_IRFs_new([model_foreign], paths,abs_value=abs_value,T_max=20, shocktitle='Demand shock', scale=True, do_sumplot=False, pathlabels=labels)

#%%

abs_value = ['pi']

paths = ['r','pi', 'i'] 
figs.show_IRFs_new([model], paths,abs_value=abs_value,T_max=30, shocktitle='Demand shock', do_sumplot=False, scale=True)


dpi = model.path.pi*model.par.scale * 100

plt.plot(dpi[0,:21])
plt.show()


dr = (model.path.r-model.ss.r)*model.par.scale * 100

plt.plot(dpi[0,:21])
plt.plot(-dr[0,:21])
plt.show()


#%%

abs_value = ['r', 'ra', 'iF_s', 'goods_mkt', 'NKPC', 'Taylor_rule', 'StockPrice', 'E', 'NFA', 'Walras', 'wnNT', 'iF_s', 'pi', 'piH', 'piF', 'UniformT', 'NFA', 'NX', 'piF_s']
# paths = ['C_s','iF_s','piF_s', 'PF_s']  
#figs.show_IRFs_new([model], paths,abs_value=abs_value,T_max=40, shocktitle='Demand shock', scale=True, do_sumplot=False)
#paths = ['Y','C','r', 'ra', 'Walras', 'A',  'Q', 'pi', 'piNT', 'piH', 'wnT', 'wnNT', 'YT', 'YNT', 'NT', 'NNT', 'B', 'LT', 'G', 'YT', 'G_T', 'G_NT', 'NX', 'PF_s', 'C_s', 'iF_s'] 
paths = ['VA','GDP', 'VA_T', 'VA_NT', 'OT', 'ONT', 'GDP_T', 'GDP_NT', 'CNT', 'CT'] 
figs.show_IRFs_new([model], paths,abs_value=abs_value,T_max=40, shocktitle='Demand shock', do_sumplot=False, scale=True)


#%%


varlist =  ['GDP', 'GDP_T', 'GDP_NT', 'VA', 'VA_T', 'VA_NT', 'C', 'CT', 'CNT',
             'XT', 'XNT', 'NT', 'NNT', 'Income', 
            'r',  'PH', 'PH_s', 'PF',  
            'Exports', 'Imports', 'E', 'Q', 'UniformT_exo',
            {'PF/PH (ToT)' : ['PF', 'PH', False]}, {'C/C_s' : ['C', 'C_s', False]}, {'P/PF_s' : ['P', 'PF_s', False]},
            {'C/GDP cumulative' : ['C', 'GDP', True]}, {'C/VA cumulative' : ['C', 'VA', True]}, {'GDP_NT/GDP_T cumulative' : ['GDP_NT', 'GDP_T', True]},
            {'CH/CNT cumulative' : ['CH', 'CNT', True]}, {'CT/CNT cumulative' : ['CT', 'CNT', True]}
            ]

scalevar = 'GDP'
upars = None
paramvals = [True, False]

fig,_ = figs.vary_irfs(model_base=model,shock='UniformT_exo', varlist=varlist, scalevar=scalevar, ncols=3, 
                     paramlist='debt_rule', paramvals=paramvals, alphacolor=False, pctp=['r'],
                     foreignshock=False,  HH_type='HA', T_max=30)


#%%

model.compute_jacs(do_print=True,skip_shocks=False,skip_hh=False)

#%%

utils.Create_shock('G_exo', model, 0.001, absval=True)

model.transition_path(do_print=True)


#%%

# for k in model.shocks:
#     #setattr(model.par,'std_'+k,0.0)
#     setattr(model.par,'jump_'+k,0.0)
    
 
model_foreign = GetForeignEcon.get_foreign_econ(shocksize=0.001)
GetForeignEcon.create_foreign_shock(model, model_foreign)
   
model.find_transition_path(do_print=True)

#%%
import seaborn as sns
sns.set_theme()
sns.set_style("whitegrid")

paths = ['C', 'Y']
model.find_IRFs(do_print=True,reuse_G_U=False,shock_specs={'dPF_s' : model.path.PF_s[0,:]-model.ss.PF_s})
model.show_IRFs(paths,ncols=3,T_max=300,abs_diff=paths,do_linear=True,do_targets=False, do_shocks=False)



#%%

test = model.par.M_Y

from numpy.linalg import inv
from numpy import linalg as LA

print(LA.eig(test))

from numpy import linalg as lg

Eigenvalues, Eigenvectors = lg.eig(test)
Lambda = np.diag(Eigenvalues)
    
test1 = Eigenvectors @ Lambda @ Eigenvectors.T
print(test1)


#%%

test2 = test @ test

print(np.allclose(test, test2))


#%%

T = 2000
Onemat = np.ones((T,T))
beta = model.par.beta_mean 

q = beta**(np.arange(T))

Mmat = Onemat @ q * (1-beta)
 
prod = Mmat @ Mmat


print(np.allclose(prod, Mmat))

#%%

temp1 = Onemat @ q.T
temp2 = q.T @ Onemat



#%%

from scipy.optimize import minimize
from utils import nonlin_MPC 

par=model.par
ss=model.ss
#par.a_min = - (ss.wnNT*(1-par.sT) + ss.wnT*par.sT)
par.a_min = - (ss.wnNT*(1-par.sT) + ss.wnT*par.sT) * 1.0
par.ra_pen = 0.06 

def LM_res(x):
        par.beta_mean, par.beta_delta = x[0], x[1] 
        #par.beta_delta = maxbeta - par.beta_mean
    
        
        model.solve_hh_ss(do_print=False)
        model.simulate_hh_ss(do_print=False)  

        ss.A = np.sum(ss.D*ss.a)
        ss.C = np.sum(ss.D*ss.c)
        
        #MPC_ann,_ = utils.print_MPCs(model, do_print=False)
        Target_first_year_MPC = True
        
        MPC_ann = nonlin_MPC(model)[0]
        if Target_first_year_MPC:            
            MPC_res = MPC_ann[0] - par.Agg_MPC

        res = abs((ss.A-ss.pD-ss.B)) + abs(MPC_res)
        #print(((ss.A-ss.pD-ss.B)) , (MPC_res))
        return np.array([(ss.A-ss.pD-ss.B), MPC_res])
    
maxbeta = 0.9998 * 1/(1+ss.r)
cons = ({'type': 'ineq', 'fun': lambda x:  -(x[0] + x[1]) + maxbeta})
bnds = ((0.8, None), (0, None))
    
x0= np.array([0.95125124, 0.04357041])
#results = minimize(LM_res, x0, method='BFGS',constraints=cons,bounds=bnds)
results=utils.broyden_solver_autojac(LM_res, x0=x0, maxcount=30, noisy=True, tol=1E-6)   
print(results)



#%%
from numpy.linalg import inv

beta =  0.98 

B_mat = np.eye(10)
B_mat -= np.diag(np.ones(10-1), 1) 

beta_mat = np.eye(10)
beta_mat -= np.diag(np.ones(10-1), 1) * beta

prod = beta_mat @ B_mat
invprod = inv(prod)

#%%

from sympy import *

beta=Symbol('beta')
sHtM = Symbol('sHtM')
sigma = Symbol('sigma')
rho = Symbol('rho')
dc0 = Symbol('dc0')


matR = Matrix([[1-beta, (1-beta)*beta, (1-beta)*beta**2], [1-beta, (1-beta)*beta, (1-beta)*beta**2], [1-beta, (1-beta)*beta, (1-beta)*beta**2]])

M_mat = matR*(1-sHtM) + sHtM*eye(3)

U = Matrix([[1, 1, 1], [0, 1, 1], [0, 0, 1]])

M_R = -1/sigma * (eye(3) - matR) * U * (1-sHtM)


bbetaprod = Matrix([[1, 1+beta, 1+beta+beta**2], [0, 1, 1+beta], [0, 0, 1]])

Mtest = M_mat * bbetaprod

#%%
dC = Matrix([[1, rho, rho**2]]).T * dc0

shock = bbetaprod * dC          
    

#%%

fig = figs.PE_MonPol_shock(model, lin=False)
fig.savefig(f'..\plots\calibration\Rshock.pdf')



#%%

fig, fig_quarterly, fig_ann = figs.plot_MPCs(model, lin=False)
fig.savefig(f'..\plots\calibration\MPCs.pdf')
fig_quarterly.savefig(f'..\plots\calibration\MPCs_quarterly.pdf')
fig_ann.savefig(f'..\plots\calibration\MPCs_ann.pdf')



#%%

#model.par.kappaH_s  = 999.0
#model.par.r_debt_elasticity = 0.0001

#model.par.etaT = 0.05


#model.par.use_RA_jac = True 

if model.par.HH_type == 'HA':
    model.compute_jacs(do_print=True,skip_shocks=True,skip_hh=False)
else:
    model.compute_jacs(do_print=True,skip_shocks=True,skip_hh=True)


#%%
#testjac = model.jac_hh[('C_hh', 'UniformT')].copy()
model._compute_jac_hh()
testjac_lage = model.jac_hh[('C_hh', 'UniformT')].copy()


dI = np.zeros(model.par.T)
dI[:] = 0.0 
dI[0] = 0.05 * (model.ss.WT * model.ss.NT + model.ss.WNT * model.ss.NNT)

dC = testjac_lage @ dI / dI[0]

plt.plot(dC[:20])
plt.show()
  
#%%
model.par.max_iter_broyden = 7
# utils.compute_jac_sym(model, h=1e-04)
# utils.compute_jac_exo_sym(model, h=1e-04)
# utils.Create_shock('UniformT_exo', model, 0.1*model.ss.Y, absval=True)
# utils.Create_shock('betaF', md, -0.0001, absval=False)
# utils.Create_shock('zT', model, 0.00, absval=True)
# utils.Create_shock('G_exo', model, 0.001, absval=True)

model.par.beta_corr = 0.0 

model_foreign = GetForeignEcon.get_foreign_econ(shocksize=0.001)

GetForeignEcon.create_foreign_shock(model, model_foreign)

# utils.scaleshock('G', model)
        
#%%

# model.path.iF_s[:,:] = model.ss.r 
# model.path.PF_s[:,:] = model.ss.PF_s 
# model.path.C_s[:,:] = model.ss.C_s 

#model.find_transition_path(shock_specs=shock_specs, do_print=True)

#utils.Create_shock('G_exo', model, 0.01, absval=True)
model.transition_path(do_print=True)

# comp.find_transition_path(model, do_print=True, use_jac_hh=True)
# utils.scaleshock('Z', model)
# utils.scaleshock('C', model)
utils.scaleshock('C_s', model)


abs_value = ['r', 'ra', 'iF_s', 'goods_mkt', 'NKPC', 'Taylor_rule', 'StockPrice', 'E', 'NFA', 'Walras', 'wnNT', 'iF_s', 'pi', 'piH', 'piF', 'UniformT', 'NFA', 'NX', 'piF_s']
# paths = ['C_s','iF_s','piF_s', 'PF_s']  
#figs.show_IRFs_new([model], paths,abs_value=abs_value,T_max=40, shocktitle='Demand shock', scale=True, do_sumplot=False)
#paths = ['Y','C','r', 'ra', 'Walras', 'A',  'Q', 'pi', 'piNT', 'piH', 'wnT', 'wnNT', 'YT', 'YNT', 'NT', 'NNT', 'B', 'LT', 'G', 'YT', 'G_T', 'G_NT', 'NX', 'PF_s', 'C_s', 'iF_s'] 
paths = ['Y','GDP','C', 'r', 'Walras', 'YT',  'YNT', 'wnNT', 'wnT', 'TradeBalance',  'C', 'CNT', 'NFA', 'Exports', 'DivT' , 'DivNT', 'Q',
          'Imports', 'NX', 'B', 'GDP', 'GDP_NT', 'GDP_T', 'XT', 'XNT', 'OT', 'ONT',  'Z'] 
figs.show_IRFs_new([model], paths,abs_value=abs_value,T_max=40, shocktitle='Demand shock', do_sumplot=False, scale=True)


#%%

abs_value = ['r', 'ra', 'iF_s', 'goods_mkt', 'NKPC', 'Taylor_rule', 'StockPrice', 'E', 'NFA', 'Walras', 'wnNT', 'iF_s', 'pi', 'piH', 'piF', 'UniformT', 'NFA', 'NX', 'piF_s']
# paths = ['C_s','iF_s','piF_s', 'PF_s']  
#figs.show_IRFs_new([model], paths,abs_value=abs_value,T_max=40, shocktitle='Demand shock', scale=True, do_sumplot=False)
#paths = ['Y','C','r', 'ra', 'Walras', 'A',  'Q', 'pi', 'piNT', 'piH', 'wnT', 'wnNT', 'YT', 'YNT', 'NT', 'NNT', 'B', 'LT', 'G', 'YT', 'G_T', 'G_NT', 'NX', 'PF_s', 'C_s', 'iF_s'] 
paths = ['VA','GDP', 'VA_T', 'VA_NT', 'OT', 'ONT', 'GDP_T', 'GDP_NT', 'CNT', 'CT'] 
figs.show_IRFs_new([model], paths,abs_value=abs_value,T_max=40, shocktitle='Demand shock', do_sumplot=False, scale=True)



#%%

ss=model.ss
path=model.path 

dPNT = path.PNT[0,:]
dPGDP = path.PGDP[0,:] 

dCNT = (path.CNT[0,:] - ss.CNT ) * model.par.scale 
dGDPNT = (path.GDP_NT[0,:]*dPGDP/dPNT - ss.GDP_NT ) * model.par.scale 

plt.plot(dCNT[:20])
plt.plot(dGDPNT[:20])
plt.show()

# PNT * (YNT - par.FixedCost[1])  PNT*XNT2T - PNT*XNT2NT = PNT * CNT + G_NT * P + INT * P 

rhs = path.YNT - model.par.FixedCost[1] - path.XNT2T - path.XNT2NT
rhsss = ss.YNT - model.par.FixedCost[1] - ss.XNT2T - ss.XNT2NT
dGDPNT = (path.GDP_NT[0,:]*dPGDP/dPNT - ss.GDP_NT ) * model.par.scale 

VA = GDP - XNT2T

plt.plot(dGDPNT[:20])
plt.plot((rhs[0,:20] - rhsss)* model.par.scale )
plt.show()


#%%

paths = ['C_s','iF_s','piF_s', 'PF_s']  

#model_foreign.show_IRFs(paths,abs_diff=['piF_s'],ncols=3,T_max=40)


# PNT * YNT - PNT*XNT2T - PNT*XNT2NT - (PNT * CNT + G_NT * P + INT * P )  

# goods_mkt_NT[:] = PNT * (YNT - par.FixedCost[1]) - PNT*XNT2T - PNT*XNT2NT - (PNT * CNT )  

# paths = ['GDP_T','GDP_NT', 'OT','ONT','CNT', 'CH', 'CT', 'CF']  

figs.show_IRFs_new([model], paths,abs_value=abs_value,T_max=40, shocktitle='Demand shock', do_sumplot=False, scale=True)


#%%

C_test = np.zeros(T)
par=model.par
C_util = model.ss.CR**(-par.sigma-1)
for tt in range(T):
    t = par.T-1-tt      
    if t == par.T-1:
        C_test[t] = model.ss.CR
    else:       
        C_test[t] = ((1+model.path.r[0,t]) * par.beta_mean * C_test[t+1]**(-par.sigma))**(-1/par.sigma)  

beta = model.par.beta_mean
T =model.par.T
jacmat = np.zeros((model.par.T,model.par.T))
U = np.triu(np.ones((T,T)),k=0)

for j in range(model.par.T):
    jacmat[j,:] = (1-beta)*beta**np.arange(model.par.T)

jacmat_R = -1/model.par.sigma * (np.eye(T) - jacmat) @ U  *beta * model.ss.CR
  

dCR_r_test = np.zeros(T)
dr = (model.path.r[0,:]   - model.ss.r) 
for tt in range(T):
    t = par.T-1-tt  
    dCR_r_test[t] = -1/model.par.sigma * np.sum(dr[t:]/(1+model.ss.r)) 

dCR_r_test1 = np.zeros(T)
dr = (model.path.r[0,:]   - model.ss.r) 
for tt in range(T):
    t = par.T-1-tt  
    if t == par.T-1:
        dCR_r_test1[t] = 0.0
    else:
        dCR_r_test1[t] = dCR_r_test1[t+1] -1/model.par.sigma *  dr[t] * beta * model.ss.CR  

dC_dI = jacmat @ (model.path.Income[0,:]   - model.ss.Income) 
dC_dr = jacmat_R @ (model.path.r[0,:]   - model.ss.r) 
dC = model.path.C[0,:] - model.ss.C
dC_test= C_test - model.ss.CR

#plt.plot(dC_dI[:40])
plt.plot(dC_test[:40])
#plt.plot(dC_test[:40])
plt.plot(dC_dr[:40], '-.')
#plt.plot(dCR_r_test[:40], '--')
plt.plot(dCR_r_test1[:40], '.')
plt.show()


#%%
M_Y = np.zeros((model.par.T,model.par.T))
U = np.triu(np.ones((T,T)),k=0)

for j in range(par.T):
    M_Y[j,:] = (1-beta)*beta**np.arange(par.T)

M_R = -1/model.par.sigma * (np.eye(T) - M_Y) @ U  * model.ss.CR * (1+ss.r) 
M_beta = -1/model.par.sigma * (np.eye(T) - jacmat) @ U  * (1+ss.r) * model.ss.CR


#%% Jac mats 


model_HA = HANKModelClass(name='baseline')
model_HA.find_ss(do_print=False)
#model_HA.compute_jacs(do_print=True,skip_shocks=True,skip_hh=False)
model_HA._compute_jac_hh(dx=0.03)

#%%

model_RA = model_HA.copy()
model_RA.par.HH_type = 'RA-IM'
model_RA.find_ss(do_print=False)

#%%

from seaborn import color_palette, set_palette
set_palette("colorblind")
plt.style.use('seaborn-white')

lsize = 1.5
Nquarters = 30 

HA_dC_dI = model_HA.jac_hh[('C_hh', 'UniformT')]
HA_dC_dr = model_HA.jac_hh[('C_hh', 'ra')]

RA_dC_dI = model_RA.par.M_Y.copy() 
RA_dC_dr = model_RA.par.M_R.copy() 
# Convert RA_dC_dr to ra dating 
RA_dC_dr[:,1:] = RA_dC_dr[:,:-1].copy()
RA_dC_dr[:,0] = RA_dC_dI[:,0] * model_RA.ss.r


columns = [0,4, 8, 12, 16, 20]
alphalist = np.flip(np.linspace(0.4,1,len(columns)))
l1,l2 = 'HANK', 'RANK'

fig = plt.figure(figsize=(8,3.0))

ax = fig.add_subplot(1,2,1)
# plt.plot(x1, y1, 'ko-')
ax.set_title(r'$\mathbf{M}$')

l1_c,l2_c = l1, l2
for col in columns:   
    ax.plot(HA_dC_dI[:Nquarters, col], '-', label=l1_c, color='C2', linewidth=lsize)
    ax.plot(RA_dC_dI[:Nquarters, col], '-', label=l2_c, color='C1', linewidth=lsize)    
    l1_c, l2_c = '_nolegend_', '_nolegend_' # suppress extra legends

ax.plot(np.zeros(Nquarters), '--', color='black')

ax.set_xlabel('Quarters', fontsize=12)
# plt.ylabel('Damped oscillation')
#ax.legend(frameon=True, fontsize=10)
ax.set_ylabel(f'$dC$', fontsize=12)


ax = fig.add_subplot(1,2,2)
ax.set_title(r'$\mathbf{M_r}$')

l1_c,l2_c = l1, l2
for i,col in enumerate(columns):   
    ax.plot(HA_dC_dr[:Nquarters, col], '-', label=l1_c, color='C2', linewidth=lsize, alpha=alphalist[i])
    ax.plot(RA_dC_dr[:Nquarters, col], '-', label=l2_c, color='C1', linewidth=lsize, alpha=alphalist[i])    
    l1_c, l2_c = '_nolegend_', '_nolegend_' # suppress extra legends

ax.plot(np.zeros(Nquarters), '--', color='black')

ax.set_xlabel('Quarters', fontsize=12)
# plt.ylabel('Damped oscillation')
ax.legend(frameon=True, fontsize=12)
ax.set_ylabel(f'$dC$', fontsize=12)


plt.tight_layout()
fig.savefig(f'..\plots\calibration\M_columns.pdf')

plt.show()

#%%


plt.plot(model.path.DomP[0,:30]-1.0)
plt.plot(DomPi_chain[:30]-1.0,'--')
plt.show()


#%%

GDPTEst =  model.path.ONT
GDPTEstss =  model.ss.ONT


GDPTEst2 = (model.path.PNT * model.path.YNT - model.path.PNT*model.path.XNT2T - model.path.PNT*model.path.XNT2NT) /model.path.PGDP
GDPTEstss2 = (model.ss.PNT * model.ss.YNT - model.ss.PNT*model.ss.XNT2T - model.ss.PNT*model.ss.XNT2NT) /model.ss.PGDP

plt.plot((GDPTEst[0,:40]-GDPTEstss)*model.par.scale/GDPTEstss*100)
plt.plot((GDPTEst2[0,:40]-GDPTEstss2)*model.par.scale/GDPTEstss2*100)
plt.show()


#%%

abs_value = ['r', 'ra', 'iF_s', 'goods_mkt', 'NKPC', 'Taylor_rule', 'StockPrice', 'E', 'NFA', 'Walras', 'wnNT', 'iF_s', 'pi', 'piH', 'piF', 'UniformT', 'NFA', 'NX']
# paths = ['C_s','iF_s','piF_s', 'PF_s']  
#figs.show_IRFs_new([model], paths,abs_value=abs_value,T_max=40, shocktitle='Demand shock', scale=True, do_sumplot=False)
#paths = ['Y','C','r', 'ra', 'Walras', 'A',  'Q', 'pi', 'piNT', 'piH', 'wnT', 'wnNT', 'YT', 'YNT', 'NT', 'NNT', 'B', 'LT', 'G', 'YT', 'G_T', 'G_NT', 'NX', 'PF_s', 'C_s', 'iF_s'] 
paths = ['Y','GDP','C', 'r', 'Walras', 'YT',  'OT', 'ONT',  'A'] 
figs.show_IRFs_new([model], paths,abs_value=abs_value,T_max=40, shocktitle='Demand shock', do_sumplot=False, scale=True)


#%%

ss=model.ss
par=model.par 


test =  (ss.PO_T/par.prodbeta[0])**par.prodbeta[0] * (ss.PXT/par.prodalpha[0])**par.prodalpha[0] / (ss.Z*ss.zT)
par.prodbeta[0]*(PH/((ss.PXT/par.prodalpha[0])**par.prodalpha[0] / (ss.Z*ss.zT)))**(1/par.prodbeta[0]) = PO_T

print(ss.PH, test)


test =  (ss.PO_NT/par.prodbeta[1])**par.prodbeta[1] * (ss.PXNT/par.prodalpha[1])**par.prodalpha[1] / (ss.Z*ss.zNT)

print(ss.PNT, test)

#%%

test = ss.PH * ss.YT - ss.PO_T*ss.OT - ss.XT*ss.PXT
print(test)

test1 =  ss.PXT*ss.XT - (ss.PNT*ss.XNT2T + ss.PH*ss.XT2T)
print(test1)

test1 =  ss.PXNT*ss.XNT - (ss.PNT*ss.XNT2NT + ss.PH*ss.XT2NT)
print(test1)


#%%

model_Copy = model.copy()
def homotopic_cont(model_, target_vars, target_vals, nsteps = 3, noisy=True, orgvals=None, use_HtM_model=True):
    if orgvals is None:
        orgvals = [getattr(model_.par,x) for x in target_vars]
    if use_HtM_model:
        model_.par.HH_type = 'TA-IM'
        for k in range(len(target_vars)):
            setattr(model_.par, target_vars[k], target_vals[k])
        model_.find_ss(do_print=True)
        model_.par.x0[0] = model_.par.FixedCost
        model_.par.x0[1] = model_.ss.zNT
        model_.par.x0[2] = model_.ss.zT
        model_.par.HH_type = 'HA'
        model_.find_ss(do_print=True)
    else:
        stpsize = 0
        for j in range(nsteps):             
            updatedvals = [orgvals[k] * (1- (1+stpsize+j)/(stpsize+nsteps)) + target_vals[k] * (1+stpsize+j)/(stpsize+nsteps) for k in range(len(target_vars))]
            if noisy: print(f'Iteration {1+j}')
            for k in range(len(target_vars)):
                setattr(model_.par, target_vars[k], updatedvals[k])
                if noisy: print(f'Parameter value for {target_vars[k]} = {updatedvals[k]:.3f}')
           
            model_.find_ss(do_print=False)
    
   
homotopic_cont(model_Copy, ['muT', 'muNT'], [1.03, 1.2], nsteps = 3, noisy=True, orgvals=None, use_HtM_model=True)        
        

#%%

model_c = HANKModelClass(name='baseline')
model_c.par.HH_type = 'HA'
model_c.find_ss(do_print=True)

#%%

par = model.par 
ss=model.ss 

exp = (par.sigma_NX-1)/par.sigma_NX
ss.YT  =  ss.Z * ss.zT*( ((1-par.alphaK-par.alphaX_T) * ss.NT**exp +  par.alphaX_T * XT**exp ))**(1/exp)
ss.YNT  = ss.Z * ss.zNT*( ((1-par.alphaK-par.alphaX_NT) * ss.NNT**exp +  par.alphaX_NT * XNT**exp ))**(1/exp)
    
ss.WT  = ss.PH *  (1-par.alphaK-par.alphaX_T) *(ss.YT/ss.NT)**(1/par.sigma_NX) / par.muT * (ss.Z * ss.zT)**((par.sigma_NX-1)/par.sigma_NX)
ss.WNT = ss.PNT * (1-par.alphaK-par.alphaX_NT) *(ss.YNT/ss.NNT)**(1/par.sigma_NX)  / par.muNT * (ss.Z * ss.zNT)**((par.sigma_NX-1)/par.sigma_NX)
    
XT = par.alphaX_T**(par.sigma_NX) * (ss.PXT*par.muT / ss.PH)**(-par.sigma_NX) * ss.YT * (ss.Z * ss.zT)**(par.sigma_NX-1)
XNT = par.alphaX_NT**(par.sigma_NX) * (ss.PXNT*par.muNT / ss.PNT)**(-par.sigma_NX) * ss.YNT * (ss.Z * ss.zNT)**(par.sigma_NX-1)

#res3 = XT*ss.PXT/(ss.WT*ss.NT + XT*ss.PXT) - par.XT_expshare
#res4 = XNT*ss.PXNT/(ss.WNT*ss.NNT + XNT*ss.PXNT) - par.XNT_expshare
res3 = XT*ss.PXT/(ss.PH*ss.YT) #- par.XT_expshare
res4 = XNT*ss.PXNT/(ss.PNT*ss.YNT) #- par.XNT_expshare

print(res3, res4)



#%%

varlist = ['YT', 'YNT', 'wnT', 'wnNT', 'C_T', 'C_NT', 'DivT', 'DivNT', 'XT', 'XNT', 'zT', 'zNT']
# varlist = ['XT', 'XNT', 'XT2NT', 'XT2T', 'LT']
for var in varlist:
    print(var)
    print(getattr(model.ss, var), getattr(model_c.ss, var))
    
#%%

utils.print_MPCs_s(model,0,do_print=True)
utils.print_MPCs_s(model,1,do_print=True)


#%%

figs.plot_MPCs(model, lin=False, plot_sectoral=True)


#%%

figs.C_decomp(model, 40, lwidth=1.5)

#%%

test = model.ss.XT*model.ss.PXT/(model.ss.YT + model.par.FixedCost*model.ss.YT)
test_NT = model.ss.XNT*model.ss.PXNT/(model.ss.YNT + model.par.FixedCost*model.ss.YNT)

print(test)
print(test_NT)


# test = model_c.ss.XT*model_c.ss.PXT/(model_c.ss.YT + model_c.par.FixedCost*model_c.ss.YT)
# test_NT = model_c.ss.XNT*model_c.ss.PXNT/(model_c.ss.YNT + model_c.par.FixedCost*model_c.ss.YNT)

# print(test)
# print(test_NT)


test = model.ss.NT*model.ss.WT/(model.ss.YT + model.par.FixedCost*model.ss.YT)
test_NT = model.ss.NNT*model.ss.WNT/(model.ss.YNT + model.par.FixedCost*model.ss.YNT)

print(test)
print(test_NT)


# test = model_c.ss.NT*model_c.ss.WT/(model_c.ss.YT + model_c.par.FixedCost*model_c.ss.YT)
# test_NT = model_c.ss.NNT*model_c.ss.WNT/(model_c.ss.YNT + model_c.par.FixedCost*model_c.ss.YNT)

# print(test)
# print(test_NT)

#%%

test = model.ss.NT*model.ss.WT/(model.ss.XT*model.ss.PXT)
test_NT = model.ss.NNT*model.ss.WNT/(model.ss.XNT*model.ss.PXNT)

print(test)
print(test_NT)


test = model_c.ss.NT*model_c.ss.WT/(model_c.ss.XT*model_c.ss.PXT)
test_NT = model_c.ss.NNT*model_c.ss.WNT/(model_c.ss.XNT*model_c.ss.PXNT)

print(test)
print(test_NT)

# test = model_c.ss.NT*model_c.ss.WT/(model_c.ss.YT + model_c.par.FixedCost*model_c.ss.YT)
# test_NT = model_c.ss.NNT*model_c.ss.WNT/(model_c.ss.YNT + model_c.par.FixedCost*model_c.ss.YNT)

# print(test)
# print(test_NT)

#%%

fig = figs.PE_MonPol_shock(model, lin=False)
fig.savefig(f'..\plots\calibration\Rshock.pdf')



#%%

fig, fig_quarterly, fig_ann = figs.plot_MPCs(model, lin=False)
fig.savefig(f'..\plots\calibration\MPCs.pdf')
fig_quarterly.savefig(f'..\plots\calibration\MPCs_quarterly.pdf')
fig_ann.savefig(f'..\plots\calibration\MPCs_ann.pdf')




#%%
from MC_simulation import sim_households

sim_dict = sim_households(model,T=2000, N=5000, burnin=1000, expT = expT, policyexp = None, sim_cont=False)

print(sim_dict['C'][500])


#%%

sim_dict = sim_households(model,T=3000, N=10000, burnin=1000, expT = expT, policyexp = None, sim_cont=False)
sim_dict_cont = sim_households(model,T=3000, N=10000, burnin=1000, expT = expT, policyexp = None, sim_cont=True)

#%%
print(sim_dict['Z'][500])
print(sim_dict_cont['Z'][500])

print(np.average(sim_dict_cont['E']))
print(np.std(sim_dict_cont['E']))



#%%
from MC_simulation import sim_households

expT = 4000
sim_dict, sim_dict_pol = sim_households(model,T=6000, N=20000, burnin=1000, expT = expT, policyexp = 'MPC', sim_cont=False)



#%%

mpcs = (sim_dict_pol['c'][expT,:] - sim_dict['c'][expT,:])/(sim_dict_pol['inc'][expT,:] - sim_dict['inc'][expT,:])
mpcs_ann = np.sum((sim_dict_pol['c'][expT:expT+4,:] - sim_dict['c'][expT:expT+4,:]),axis=0)/(sim_dict_pol['inc'][expT,:] - sim_dict['inc'][expT,:])

a_measure = sim_dict['a'][expT,:]/(sim_dict['inc'][expT,:] * 4 )
print(np.average(mpcs))
print(np.average(mpcs_ann))

fig = plt.figure(figsize= (4.4*1,2.9*1),dpi=100)
ax = fig.add_subplot(1,1,1)

for i_beta in range(model.par.Nbeta):
    parbeta = round(model.par.beta_grid[i_beta]*model.par.beta_grid[0],2)
    a_measure_ = a_measure[sim_dict_pol['beta'] == i_beta]
    mpcs_ = mpcs[sim_dict_pol['beta'] == i_beta]
    order = np.argsort(a_measure_[:])
    ax.plot(a_measure_[order], mpcs_[order], alpha=0.4, label=r'$\beta=$'+f'{parbeta}')
ax.set_xlim([-0.1,4])
ax.set_ylim([0,1])
ax.set_xlabel('Wealth-income ratio', fontsize=12)
ax.set_ylabel('Quarterly MPCs', fontsize=12)

ax.axhline(y=np.average(mpcs), color='black', linestyle='-', label='Average MPC', linewidth=1.4)
ax.legend(frameon=True)
plt.tight_layout()
#fig.savefig(f'plots\calibration\MPCs_dist.pdf')
plt.show()




#%%

fig = plt.figure(figsize= (4.4*1,2.9*1),dpi=100)
ax = fig.add_subplot(1,1,1)

for i_beta in range(model.par.Nbeta):
    parbeta = round(model.par.beta_grid[i_beta],2)
    mpcs_ = mpcs[sim_dict_pol['beta'] == i_beta]
    mpcs_ = mpcs_ann[sim_dict_pol['beta'] == i_beta]
    print(np.average(mpcs_))
    ax.hist( mpcs_, alpha=0.6, label=r'$\beta=$'+f'{parbeta}', bins=300, density=True)
ax.set_xlim([0,1])
ax.set_ylim([0,20])
ax.set_xlabel('Annual MPCs', fontsize=12)
ax.set_ylabel('Density', fontsize=12)

ax.axvline(x=np.average(mpcs_ann), color='black', linestyle='-', label='Average MPC', linewidth=1.4)

ax.legend(frameon=True, fontsize=9, loc='upper right')
plt.tight_layout()
fig.savefig(f'..\plots\calibration\MPCs_dist.pdf')
plt.show()

#%%




#%%
test = sim_dict['a'][3000,:]
plt.hist( test[sim_dict['beta'] == 2])
plt.show()




#%%

test = np.sum(model.sim.D[ model.sol.a <= 1e-08] )

print(100*test)

#%%

MPC = np.zeros(model.sim.D.shape)
MPC[:,:,:-1] = (model.sol.c[:,:,1:]-model.sol.c[:,:,:-1])/( (1+model.ss.r)*model.par.a_grid[np.newaxis,np.newaxis,1:]-(1+model.ss.r)*model.par.a_grid[np.newaxis,np.newaxis,:-1])
MPC[:,:,-1] = MPC[:,:,-2] # assuming constant MPC at end

#MPC = np.sum(model.sim.D*MPC,axis=0)

fig = plt.figure(figsize= (4.4*1,2.9*1),dpi=100)
ax = fig.add_subplot(1,1,1)

for i_beta in range(model.par.Nbeta):
    parbeta = round(model.par.beta_grid[i_beta]*model.par.beta_grid_s[0],2)
    ax.hist( MPC[i_beta,...], alpha=0.6, label=r'$\beta=$'+f'{parbeta}', bins=300, density=True, weights=model.sim.D[i_beta,...])
#ax.set_xlim([0,1])
#ax.set_ylim([0,30])
ax.set_xlabel('Quarterly MPCs', fontsize=12)
ax.set_ylabel('Density', fontsize=12)

ax.axvline(x=np.average(mpcs), color='black', linestyle='-', label='Average MPC', linewidth=1.4)
ax.legend(frameon=True, fontsize=10, loc='upper right')
plt.tight_layout()
#fig.savefig(f'plots\calibration\MPCs_dist.pdf')
plt.show()

#%%

from mpl_toolkits.mplot3d import Axes3D  

fig = plt.figure(figsize= (4.4*1,2.9*1),dpi=100)
ax = fig.add_subplot(111, projection='3d')

MPC = np.zeros(model.sim.D.shape)
MPC[:,:,:-1] = (model.sol.c[:,:,1:]-model.sol.c[:,:,:-1])/( (1+model.ss.r)*model.par.a_grid[np.newaxis,np.newaxis,1:]-(1+model.ss.r)*model.par.a_grid[np.newaxis,np.newaxis,:-1])
MPC[:,:,-1] = MPC[:,:,-2] # assuming constant MPC at end

#MPC = np.sum(model.sim.D*MPC,axis=0)

X, Y = np.meshgrid(model.par.z_grid_ss, model.par.a_grid)
ax.plot_surface(X, Y, MPC.T)

#ax.set_xlim([-0.1,12])
#ax.set_ylim([0,1])
ax.set_xlabel('Wealth-income ratio', fontsize=12)
ax.set_ylabel('Quarterly MPCs', fontsize=12)

ax.axhline(y=np.average(mpcs), color='black', linestyle='-', label='Average', linewidth=1.4)
ax.legend(frameon=True)
plt.tight_layout()
#fig.savefig(f'plots\calibration\MPCs_dist.pdf')
plt.show()


#%%

MPC = np.zeros(model.sim.D.shape)
MPC[:,:,:-1] = (model.sol.c[:,:,1:]-model.sol.c[:,:,:-1])/( (1+model.ss.r)*model.par.a_grid[np.newaxis,np.newaxis,1:]-(1+model.ss.r)*model.par.a_grid[np.newaxis,np.newaxis,:-1])
MPC[:,:,-1] = MPC[:,:,-2] # assuming constant MPC at end

MPC = np.sum(model.sim.D*MPC,axis=1)


fig = plt.figure(figsize= (4.4*1,2.9*1),dpi=100)
ax = fig.add_subplot(1,1,1)

#a_measure = sim_dict['a']/(sim_dict['inc'] * 4 )
#order = np.argsort(a_measure[expT,:])
ax.plot(model.par.a_grid,MPC.T)
ax.set_xlim([-0.1,4])
ax.set_ylim([0,1])
ax.set_xlabel('Wealth-income ratio', fontsize=12)
ax.set_ylabel('Quarterly MPCs', fontsize=12)

ax.axhline(y=np.average(mpcs), color='black', linestyle='-', label='Average', linewidth=1.4)
ax.legend(frameon=True)
plt.tight_layout()
#fig.savefig(f'plots\calibration\MPCs_dist.pdf')
plt.show()


#%%

expT = sim_dict['info']['expT']
#mpc = (sim_dict_pol['c'][expT-10:expT+10,:] - sim_dict['c'][expT-10:expT+10,:]) /(0.33*sim_dict['z'][expT,:])
mpc = (sim_dict_pol['c'][expT:expT+10,:] - sim_dict['c'][expT:expT+10,:]) /(0.33*sim_dict['z'][expT,:])

MPC = np.average(mpc,axis=1)

plt.plot(MPC)
plt.show()

#%%

model_foreign = GetForeignEcon.get_foreign_econ(shocksize=0.001,shock='iF_s_exo')
GetForeignEcon.create_foreign_shock(model, model_foreign)

#%%

abs_value = ['r', 'ra', 'iF_s', 'rF_s', 'NKPC', 'Taylor_rule', 'StockPrice', 'E', 'NFA', 'Walras', 'wnNT', 'iF_s', 'pi', 'piH', 'piF', 'UniformT', 'NFA', 'NX']
paths = ['C_s','iF_s', 'iF_s_exo', 'PF_s', 'piF_s', 'rF_s', 'betaF', 'ZF']  
#figs.show_IRFs_new([model], paths,abs_value=abs_value,T_max=40, shocktitle='Demand shock', scale=True, do_sumplot=False)
#paths = ['Y','C','r', 'ra', 'Walras', 'A',  'Q', 'pi', 'piNT', 'piH', 'wnT', 'wnNT', 'YT', 'YNT', 'NT', 'NNT', 'B', 'LT', 'G', 'YT', 'G_T', 'G_NT', 'NX', 'PF_s', 'C_s', 'iF_s'] 


figs.show_IRFs_new([model_foreign], paths,abs_value=abs_value,T_max=40, shocktitle='Demand shock', do_sumplot=False, scale=False)


#%%
#model.par.TaylorType = 'NT' 
#model.par.TaylorType = 'DomP'
# model.par.TaylorType = 'NT'

# model.par.UseTaylor = True

# model.par.TaylorType = 'CPI'

# model.compute_jac(do_print=False)
model.par.debt_rule=False 
if model.par.HH_type == 'HA':
    model.compute_jacs(do_print=True,skip_shocks=True,skip_hh=False)
else:
    model.compute_jacs(do_print=True,skip_shocks=True,skip_hh=True)
# comp.compute_jac_exo(model)
# model.check_determinancy()



#%%
model.par.max_iter_broyden = 10
# utils.compute_jac_sym(model, h=1e-04)
# utils.compute_jac_exo_sym(model, h=1e-04)
# utils.Create_shock('UniformT_exo', model, 0.1*model.ss.Y, absval=True)
# utils.Create_shock('betaF', md, -0.0001, absval=False)
# utils.Create_shock('zT', model, 0.00, absval=True)


model_foreign = GetForeignEcon.get_foreign_econ(shocksize=0.001)

GetForeignEcon.create_foreign_shock(model, model_foreign)


#%%
# model.path.iF_s[:,:] = model.ss.r 
# model.path.PF_s[:,:] = model.ss.PF_s 
# model.path.C_s[:,:] = model.ss.C_s 

#model.find_transition_path(shock_specs=shock_specs, do_print=True)

#utils.Create_shock('G_exo', model, 0.01, absval=True)
model.transition_path(do_print=True)

# comp.find_transition_path(model, do_print=True, use_jac_hh=True)
# utils.scaleshock('zT', model)
# utils.scaleshock('C', model)
utils.scaleshock('C_s', model)


#%%


abs_value = ['r', 'ra', 'iF_s', 'goods_mkt', 'NKPC', 'Taylor_rule', 'StockPrice', 'E', 'NFA', 'Walras', 'wnNT', 'iF_s', 'pi', 'piH', 'piF', 'UniformT', 'NFA', 'NX']
paths = ['C_s','iF_s','PF_s']  
#figs.show_IRFs_new([model], paths,abs_value=abs_value,T_max=40, shocktitle='Demand shock', scale=True, do_sumplot=False)
#paths = ['Y','C','r', 'ra', 'Walras', 'A',  'Q', 'pi', 'piNT', 'piH', 'wnT', 'wnNT', 'YT', 'YNT', 'NT', 'NNT', 'B', 'LT', 'G', 'YT', 'G_T', 'G_NT', 'NX', 'PF_s', 'C_s', 'iF_s'] 


figs.show_IRFs_new([model], paths,abs_value=abs_value,T_max=500, shocktitle='Demand shock', do_sumplot=False, scale=True)



#%%
abs_value = ['r', 'ra', 'iF_s', 'goods_mkt', 'NKPC', 'Taylor_rule', 'StockPrice', 'E', 'NFA', 'Walras', 'wnNT', 'iF_s', 'pi', 'piH', 'piF', 'UniformT', 'NFA', 'NX']
#paths = ['C_s','iF_s','piF']  
#figs.show_IRFs_new([model], paths,abs_value=abs_value,T_max=40, shocktitle='Demand shock', scale=True, do_sumplot=False)
#paths = ['Y','C','r', 'ra', 'Walras', 'A',  'Q', 'pi', 'piNT', 'piH', 'wnT', 'wnNT', 'YT', 'YNT', 'NT', 'NNT', 'B', 'LT', 'G', 'YT', 'G_T', 'G_NT', 'NX', 'PF_s', 'C_s', 'iF_s'] 
paths = ['Y','GDP','C', 'r', 'Walras', 'YT',  'YNT', 'wnNT', 'wnT', 'TradeBalance',  'C', 'C_NT', 'NFA', 'Exports', 'DivT' , 'DivNT', 'Q',
         'Imports', 'NX', 'B', 'GDP', 'GDP_NT', 'GDP_T', 'XT', 'XNT', 'OT', 'ONT', 'G_exo'] 
figs.show_IRFs_new([model], paths,abs_value=abs_value,T_max=120, shocktitle='Demand shock', do_sumplot=False, scale=True)



#%%

def get_elasticity(model):
    path = model.path
    ss = model.ss    
    
    dRelP = (path.PH_s[0,:]/path.PF_s[0,:])/ss.PH_s/ss.PF_s - 1
    dC = path.C_s[0,:]/ss.C_s-1
    T_el  = np.abs((path.CH_s[0,:]/ss.CH_s-1 - dC) /  dRelP)
    T_el[np.isclose(dRelP,0)] = np.nan 
    return T_el,dRelP

T_el,dRelP = get_elasticity(model)
# CH_s = alpha * (PH_s/PF_s)**(-gamma) * C_s

plt.plot(dRelP[:30])
plt.show()

plt.plot(T_el[:30])
plt.show()


print(T_el[0])

num = (1-0.95)*(1-0.999*0.95) * 3

print(num)


#%%

model_foreign = GetForeignEcon.get_foreign_econ()
GetForeignEcon.create_foreign_shock(model, model_foreign)

#%%
from statsmodels.tsa.ar_model import AutoReg
from copy import deepcopy
#def smooth_foreign_shocks(model):
    
fshocklist = ['C_s', 'PF_s', 'iF_s']  
ar_order = [1,2,3]  
for shocknum,shock in enumerate(fshocklist): 
    
    zerofrom = 20
    
    print(shock)
    ar_order_ = ar_order[shocknum]
    dShock = deepcopy(getattr(model.path,shock)[0,:] - getattr(model.ss,shock))
    dShock[20:] = 0 
    mod = AutoReg(dShock, ar_order_, old_names=False)
    res = mod.fit()
    
    fittedvals = np.zeros(model.par.transition_T) * np.nan 
    fittedvals[0] = dShock[0]
    for t in range(model.par.transition_T-1):
        if ar_order_ == 1:
            fittedvals[t+1] = res.params[1] * fittedvals[t]
        elif ar_order_ ==2:
            if t>0:
                fittedvals[t+1] = res.params[1] * fittedvals[t] + res.params[2] * fittedvals[t-1]
            else:
                fittedvals[t+1] = res.params[1] * fittedvals[t]
        elif ar_order_ ==3:
            if t>1:
                fittedvals[t+1] = res.params[1] * fittedvals[t] + res.params[2] * fittedvals[t-1] + res.params[3] * fittedvals[t-2]
            elif t == 0:
                fittedvals[t+1] = res.params[1] * fittedvals[t]    
            elif t == 1:
                fittedvals[t+1] = res.params[1] * fittedvals[t]  + res.params[2] * fittedvals[t-1]     
    
    plt.plot(dShock[:50])
    plt.plot(fittedvals[:50],'--')
    plt.show()


model_foreign = GetForeignEcon.get_foreign_econ()
GetForeignEcon.create_foreign_shock(model, model_foreign)

#%%

model_foreign = GetForeignEcon.get_foreign_econ()
GetForeignEcon.create_foreign_shock(model, model_foreign)

# model.path.C_s[:,:] = model.ss.C_s
model.path.iF_s[:,:] = model.ss.iF_s
# model.path.PF_s[:,:] = model.ss.PF_s  
#utils.Create_shock('G_exo', model, 0.001, absval=True)
# utils.Create_shock('SupplySF, model, 0.001)


model.par.max_iter_broyden = 10

# model.path.G_eps[0,:] = model.ss.G_eps 
# model.path.G_eps[0,40:80] = model.ss.G_eps + 0.001 * model.ss.G_eps * 0.5**(np.arange(40))
# utils.Create_shock('G_exo', model, 0.001, absval=True)

model.find_transition_path(do_print=True)
# comp.find_transition_path(model, do_print=True, use_jac_hh=False)
utils.scaleshock('C_s', model)
# utils.scaleshock('PF_s', model)

#%%
model_foreign = GetForeignEcon.get_foreign_econ()
GetForeignEcon.create_foreign_shock(model, model_foreign)

# utils.scaleshock('G_eps', model)
utils.scaleshock('C_s', model)



#model_foreign = GetForeignEcon.get_foreign_econ()
#GetForeignEcon.create_foreign_shock(model, model_foreign)


paths = ['C_s','iF_s','piF_s', 'PF_s']  
pctp = ['iF_s', 'piF_s']
figs.show_IRFs_new([model], paths,T_max=30, shocktitle='Demand shock', scale=True, do_sumplot=False,pctp=pctp)




#%%
abs_value = ['r', 'ra', 'iF_s', 'goods_mkt', 'NKPC', 'Taylor_rule', 'StockPrice', 'E', 'NFA', 'Walras', 'wnNT', 'iF_s', 'pi', 'piH', 'piF', 'UniformT', 'NFA', 'NX']
#paths = ['C_s','iF_s','piF']  
#figs.show_IRFs_new([model], paths,abs_value=abs_value,T_max=40, shocktitle='Demand shock', scale=True, do_sumplot=False)
#paths = ['Y','C','r', 'ra', 'Walras', 'A',  'Q', 'pi', 'piNT', 'piH', 'wnT', 'wnNT', 'YT', 'YNT', 'NT', 'NNT', 'B', 'LT', 'G', 'YT', 'G_T', 'G_NT', 'NX', 'PF_s', 'C_s', 'iF_s'] 
paths = ['Y','C','r', 'ra', 'Walras', 'YT',  'YNT', 'wnNT', 'wnT', 'TradeBalance',  'C_T', 'C_NT', 'NFA', 'Exports', 'DivT' , 'DivNT', 'E',
         'Imports', 'NX', 'P', 'B', 'PF_s', 'CH_s', 'iF_s'] 
figs.show_IRFs_new([model], paths,abs_value=abs_value,T_max=40, shocktitle='Demand shock', do_sumplot=False, scale=True)






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



