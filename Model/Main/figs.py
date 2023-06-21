import matplotlib.pyplot as plt
import numpy as np
import utils
from IHANKModel import HANKModelClass
import GetForeignEcon
from scipy.interpolate import CubicSpline
from copy import deepcopy
from GEModelTools import lag, simulate_hh_path 
from matplotlib.ticker import FormatStrFormatter

import sys 
sys.path.insert(0,'..')

import matplotlib.pyplot as plt   

try:
    execfile("../figsettings.py") # Work for now, but bad practice 
except:
    execfile("figsettings.py") 


def plot_MPCs(model, lin=True, plot_sectoral=False, plot_ricardian=True, show_all=True, do_ann_mpc=True):

    # MPCs in data 
    MPC_data = {}
    MPC_data['x'] = np.arange(6)
    MPC_data['y'] =  np.array([0.554, 0.195, 0.124, 0.085, 0.064, 0.057])
    
    cs = CubicSpline(MPC_data['x'] , MPC_data['y'] )
    Nquarters = 11 
    
    MPC_data['x_int'] = np.arange(Nquarters)/4
    MPC_data['x_int_Q'] = np.arange(Nquarters)
    
    y_int = cs(MPC_data['x_int'])/4
    MPC_data['y_int'] = y_int / np.sum(y_int[:4]) * MPC_data['y'][0] 

    # Model MPCs
    dI = np.zeros(model.par.T)
    dI[0] = model.ss.wnT  / 3 # One month labor income 
    
    if lin:
        model.compute_jac_hh(do_print=False)
        dCT = model.jac_hh.C_T_UniformT @ dI  / model.par.sT
        dCNT = model.jac_hh.C_NT_UniformT @ dI  / (1-model.par.sT)   
        dCAgg = model.jac_hh.C_UniformT @ dI  
    else:
        path_org = deepcopy(model.path)
        ann_dCAgg, dCAgg, dCT, dCNT = utils.nonlin_MPC(model)
        model.path = path_org

    ann_dCT = np.zeros(round(model.par.T/4))
    ann_dCNT = np.zeros(round(model.par.T/4))
    ann_dCAgg = np.zeros(round(model.par.T/4))
    for j in range(round(model.par.T/4)):
        ann_dCAgg[j] = np.sum(dCAgg[j*4:(1+j)*4])         
        if plot_sectoral:
            ann_dCNT[j] = np.sum(dCNT[j*4:(1+j)*4])  
            ann_dCT[j] = np.sum(dCT[j*4:(1+j)*4]) 
    
    # Ricardian agent:
    beta = 1/(1+model.ss.ra)
    MPC_R = (1-(beta*(1+model.ss.r)**(1/model.par.CRRA))/(1+model.ss.r))*beta**np.arange(model.par.T)
    ann_MPC_R = np.zeros(round(model.par.T/4))
    for j in range(round(model.par.T/4)):
        ann_MPC_R[j] = np.sum(MPC_R[j*4:(1+j)*4])   

    lsize = 1.8
    fig = plt.figure(figsize=(4.4*1,2.9*2))
    
    if show_all:
        ax = fig.add_subplot(2,1,1)
        ax.set_title('Quarterly MPCs')
        ax.plot(dCAgg[:Nquarters], '-', label='HANK', color='C2', linewidth=lsize)
        ax.plot(np.zeros(Nquarters), '--', color='black')
        if plot_sectoral:
            ax.plot((dCT[:Nquarters]), '-.', label='T')
            ax.plot((dCNT[:Nquarters]), '--', label='NT')
        ax.plot(MPC_data['y_int'], linestyle='None', marker='o', label='Fagereng et al. (2021)', color='C0')
        if plot_ricardian:
            ax.plot(MPC_R[:Nquarters], '-', label='RANK', color='C1', linewidth=lsize)
        ax.set_xlabel('Quarters', fontsize=16)
        ax.legend(frameon=True, fontsize=10)
        ax.set_ylabel('MPC', fontsize=12)
        
        ax = fig.add_subplot(2,1,2)
        ax.plot(np.zeros(6), '--', color='black')
        ax.plot((ann_dCAgg[:6]), '-', label='HANK', color='C2', linewidth=lsize)
        if plot_sectoral:
            ax.plot((ann_dCT[:6]), '-.', label='T')
            ax.plot((ann_dCNT[:6]), '--', label='NT')
        ax.plot(MPC_data['y'], linestyle='None', marker='o', label='Data', color='C0')
        ax.set_title('Annual MPCs')
        ax.set_xlabel('Years', fontsize=16)
        ax.set_ylabel('MPC', fontsize=12)
        if plot_ricardian:
            ax.plot(ann_MPC_R[:6], '-', label='RANK', color='C1', linewidth=lsize)
            
        
        fig.tight_layout()

    
    
        sizetupple = (4.4*1,2.9*1)
        sepfig1 = plt.figure(figsize=sizetupple)
        ax = sepfig1.add_subplot(1,1,1)
        ax.plot(dCAgg[:Nquarters], '-', label='HANK', color='C2', linewidth=lsize)
        ax.plot(np.zeros(Nquarters), '--', color='black')
        if plot_sectoral:
            ax.plot((dCT[:Nquarters]), '-.', label='T')
            ax.plot((dCNT[:Nquarters]), '--', label='NT')
        ax.plot(MPC_data['y_int'], linestyle='None', marker='o', label='Fagereng et al. (2021)', color='C0')
        if plot_ricardian:
            ax.plot(MPC_R[:Nquarters], '-', label='RANK', color='C1', linewidth=lsize)
        ax.set_xlabel('Quarters', fontsize=16)
        ax.legend(frameon=True, fontsize=10)
        ax.set_ylabel('Quarterly MPC', fontsize=12)    
        sepfig1.tight_layout()
    
    if do_ann_mpc:
        sizetupple = (4.4*1,2.9*1)
        sepfig2 = plt.figure(figsize=sizetupple)
        ax = sepfig2.add_subplot(1,1,1)
        ax.plot(np.zeros(6), '--', color='black')
        ax.plot(MPC_data['y'], linestyle='None', marker='o', label='Fagereng et al. (2021)', color='C0')
        ax.plot((ann_dCAgg[:6]), '-', label='HANK', color='C2', linewidth=lsize)
        if plot_sectoral:
            ax.plot((ann_dCT[:6]), '-.', label='T')
            ax.plot((ann_dCNT[:6]), '--', label='NT')
        ax.set_xlabel('Years', fontsize=16)
        ax.set_ylabel('Annual MPC', fontsize=12)
        ax.set_xlim([0, 6-1])
        
        if plot_ricardian:
            ax.plot(ann_MPC_R[:6], '-', label='RANK', color='C1', linewidth=lsize)
        ax.legend(frameon=True, fontsize=12)
        sepfig2.tight_layout()    
        
    if do_ann_mpc:
        return sepfig2 
    elif show_all:
        return fig, sepfig1, sepfig2
    else:
        return fig 
    
def C_decomp(model, T_max, lwidth, testplot=False):   
    scale=True
    if scale:
        scaleval = getattr(model.par,'scale')
    else:
        scaleval = 1 
    assert model.par.HH_type == 'HA'
        
    C_decomp = {}
    C_decomp = {'C_NT' : {}, 'C_T' : {}}
    

    s_input = {'C_NT' : 'wnNT', 'C_T' : 'wnT'}
    ncols,nrows = 2,1
    set_palette("colorblind")
    fig = plt.figure(figsize=(4.3*ncols,3.6*nrows))

    
    for i,var in enumerate(['C_NT', 'C_T']):
        ax = fig.add_subplot(nrows,ncols,i+1)
        ss_C = getattr(model.ss, var)
        C_decomp[var]['Labor income'] = utils.Get_single_HA_IRF(model, s_input[var], var, scaleval)
        test = C_decomp[var]['Labor income']
        ax.plot(np.arange(T_max),C_decomp[var]['Labor income'][:T_max],label='Labor income', linewidth=lwidth)
        C_decomp[var]['Taxes'] = utils.Get_single_HA_IRF(model, 'LT', var, scaleval)
        test += C_decomp[var]['Taxes']
        ax.plot(np.arange(T_max),C_decomp[var]['Taxes'][:T_max],label='Taxes', linewidth=lwidth, color='Darkgreen')
        
        dX   = getattr(model.path, 'r')[0,:]
        X_ss = getattr(model.ss, 'r')
        X_jac_hh = getattr(model.jac_hh, var + '_' + 'ra')
        C_decomp[var]['r'] = X_jac_hh @ (dX-X_ss)*scaleval / ss_C * 100   
        ax.plot(C_decomp[var]['r'][:T_max], label='Interest rate', linewidth=lwidth)
        test += C_decomp[var]['r']
        
        dra   = getattr(model.path, 'ra')[0,:]
        dr = np.zeros_like(dra) + getattr(model.ss, 'r')
        dr[1:] =  getattr(model.path, 'r')[0,1:]
        X_jac_hh = getattr(model.jac_hh, var + '_' 'ra')
        C_decomp[var]['ra'] = X_jac_hh @ (dra-dr)*scaleval / ss_C * 100   
        ax.plot(C_decomp[var]['ra'][:T_max], label='Revaluation', linewidth=lwidth)
        test += C_decomp[var]['ra']

        
        varpath = getattr(model.path, var)[0,:]
        ax.plot((varpath[:T_max]/ss_C-1)*scaleval*100, label='Total', linewidth=lwidth, marker='.')
        ax.plot(np.zeros(T_max), '-', color='black')

        if testplot:
            ax.plot(test[:T_max], '--', label='Test', color='black')

        
        ax.set_ylabel('% diff. to s.s.')
        ax.set_xlabel('Quarters')
        if var == 'C_T':
            ax.set_title('Households in tradeable')
        else:
            ax.set_title('Households in non-tradeable')
    ax.legend(loc="best", frameon=True)  
    fig.tight_layout()
    
    return fig


def C_decomp_v2(model, T_max, lwidth, testplot=False):   
    scale=True
    if scale:
        scaleval = getattr(model.par,'scale')
    else:
        scaleval = 1 
    assert model.par.HH_type == 'HA'
        
    C_decomp = {}
    C_decomp = {'C_NT' : {}, 'C_T' : {}}
    
    s_input = {'C_NT' : 'wnNT', 'C_T' : 'wnT'}
    ncols,nrows = 2,2
    set_palette("colorblind")
    fig = plt.figure(figsize=(4.3*ncols,3.6*nrows))
    
    for i,var in enumerate(['C_NT', 'C_T']):
        var_hh = var + '_hh'
        ax = fig.add_subplot(nrows,ncols,i+1)
        ss_C = getattr(model.ss, var)
        C_decomp[var]['Labor income'] = utils.Get_single_HA_IRF(model, s_input[var], var_hh, scaleval)
        test = C_decomp[var]['Labor income']
        ax.plot(np.arange(T_max),C_decomp[var]['Labor income'][:T_max],label='Labor income', linewidth=lwidth)
        C_decomp[var]['Taxes'] = utils.Get_single_HA_IRF(model, 'LT', var_hh, scaleval)
        test += C_decomp[var]['Taxes']
        ax.plot(np.arange(T_max),C_decomp[var]['Taxes'][:T_max],label='Taxes', linewidth=lwidth, color='Darkgreen')
        
        dX   = getattr(model.path, 'r')[0,:]
        X_ss = getattr(model.ss, 'r')
  
        X_jac_hh = model.jac_hh[(var+'_hh','ra')]  
        C_decomp[var]['r'] = X_jac_hh @ (dX-X_ss)*scaleval / ss_C * 100   
        ax.plot(C_decomp[var]['r'][:T_max], label='Interest rate', linewidth=lwidth)
        test += C_decomp[var]['r']
        
        dra   = getattr(model.path, 'ra')[0,:]
        dr = np.zeros_like(dra) + getattr(model.ss, 'r')
        dr[1:] =  getattr(model.path, 'r')[0,1:]
        X_jac_hh = model.jac_hh[(var+'_hh','ra')]  
        C_decomp[var]['ra'] = X_jac_hh @ (dra-dr)*scaleval / ss_C * 100   
        ax.plot(C_decomp[var]['ra'][:T_max], label='Revaluation', linewidth=lwidth)
        test += C_decomp[var]['ra']

        
        varpath = getattr(model.path, var)[0,:]
        ax.plot((varpath[:T_max]/ss_C-1)*scaleval*100, label='Total', linewidth=lwidth, marker='.')
        ax.plot(np.zeros(T_max), '-', color='black')

        if testplot: # test that sum of individual input paths sum to total  
            ax.plot(test[:T_max], '--', label='Test', color='black')
       
        ax.set_ylabel('% diff. to s.s.')
        ax.set_xlabel('Quarters')
        if var == 'C_T':
            ax.set_title('Households in tradeable')
        else:
            ax.set_title('Households in non-tradeable')
    ax.legend(loc="best", frameon=True)  
    
    
    return fig    


def C_decomp_HA_v_RA(modellist, T_max, lwidth, testplot=False, disp_income=True, scale=True):   

    for model_ in modellist:
        if model_.par.HH_type == 'HA':
            model_HA = model_
        elif model_.par.HH_type == 'RA-IM':
            model_RA = model_
        else:
            raise ValueError('HH type need to be either HA or RA-IM')

    if scale:
        scaleval = getattr(model_HA.par,'scale')
    else:
        scaleval = 1 
        
    C_decomp = {}
    C_decomp['HA'] = {}
    ss_C = getattr(model_HA.ss, 'C')
    C_decomp['HA']['Total'] = utils.get_dX('C', model_HA, scaleval=scaleval) *100
    
    C_decomp['HA']['Labor income'] = utils.Get_single_HA_IRF(model_HA, 'wnNT', 'C_hh', scaleval) + utils.Get_single_HA_IRF(model_HA, 'wnT', 'C_hh', scaleval)
    C_decomp['HA']['test'] = deepcopy(C_decomp['HA']['Labor income'])

    C_decomp['HA']['Taxes'] = utils.Get_single_HA_IRF(model_HA, 'LT', 'C_hh', scaleval)
    C_decomp['HA']['test'] += C_decomp['HA']['Taxes']
    
    X_ss = getattr(model_HA.ss, 'ra')
    dX   = getattr(model_HA.path, 'ra')[0,:]  - X_ss
    dX[0] = 0 
    X_jac_hh = model_HA.jac_hh[('C_hh','ra')]
    C_decomp['HA']['r'] = X_jac_hh @ dX*scaleval / ss_C * 100   
    C_decomp['HA']['test'] += C_decomp['HA']['r']
    
    dra   = getattr(model_HA.path, 'ra')[0,:] - X_ss
    dra[1:] = 0 
    X_jac_hh = model_HA.jac_hh[('C_hh','ra')]
    C_decomp['HA']['ra'] = X_jac_hh @ dra*scaleval / ss_C * 100   
    C_decomp['HA']['test'] += C_decomp['HA']['ra']    

    
    # RA 
    C_decomp['RA'] = {}
    ss_C = getattr(model_RA.ss, 'C')
    C_decomp['RA']['Total'] = utils.get_dX('C', model_RA, scaleval=scaleval) *100

    sT = model_RA.par.sT
    dI = utils.get_dX('wnNT', model_RA, scaleval=scaleval, absvalue=True)*(1-sT) + utils.get_dX('wnT', model_RA, scaleval=scaleval, absvalue=True)*sT
    C_decomp['RA']['Labor income'] = model_RA.par.M_Y @  dI / ss_C * 100 
    C_decomp['RA']['test'] = deepcopy(C_decomp['RA']['Labor income'])

    C_decomp['RA']['Taxes'] = model_RA.par.M_Y @ utils.get_dX('LT', model_RA, scaleval=scaleval, absvalue=True) / ss_C * 100 
    C_decomp['RA']['test'] += C_decomp['RA']['Taxes']    

    X_ss = getattr(model_RA.ss, 'r')
    dX   = getattr(model_RA.path, 'r')[0,:]  - X_ss
    C_decomp['RA']['r'] = model_RA.par.M_R @ dX*scaleval / ss_C * 100   
    C_decomp['RA']['test'] += C_decomp['RA']['r']

    dra   = getattr(model_RA.path, 'ra')[0,:] - X_ss
    dra[1:] = 0 
    C_decomp['RA']['ra'] = model_RA.par.M_Y @ dra*scaleval / ss_C * 100   
    C_decomp['RA']['test'] += C_decomp['RA']['ra']    


    ncols,nrows = 2,1
    set_palette("colorblind")
    fig = plt.figure(figsize=(4.3*ncols,3.6*nrows))
    
    for i,hh in enumerate(['HA', 'RA']):
        ax = fig.add_subplot(nrows,ncols,i+1)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.set_xticks(np.arange(0, T_max, 4))
        ax.plot(np.zeros(T_max), '-', color='black')   
        if disp_income:
            dY = C_decomp[hh]['Labor income'][:T_max] + C_decomp[hh]['Taxes'][:T_max]
            ax.plot(np.arange(T_max),dY,label='Disp. income', linewidth=lwidth, linestyle='--')
        else:
            ax.plot(np.arange(T_max),C_decomp[hh]['Labor income'][:T_max],label='Labor income', linewidth=lwidth, linestyle='--')
            ax.plot(np.arange(T_max),C_decomp[hh]['Taxes'][:T_max],label='Taxes', linewidth=lwidth, linestyle=':')
        ax.plot(np.arange(T_max),C_decomp[hh]['r'][:T_max], label='Interest rate', linewidth=lwidth, linestyle='-.', color='C2')
        ax.plot(np.arange(T_max),C_decomp[hh]['ra'][:T_max], label='Revaluation', linewidth=lwidth, color='Firebrick', marker=',')     
        ax.plot(np.arange(T_max),C_decomp[hh]['Total'][:T_max], label='Total', linewidth=lwidth, marker='o', color='C3')
         
        ax.set_xlim([0, T_max-1])

        if testplot:
            ax.plot(np.arange(T_max),C_decomp[hh]['test'][:T_max], '--', label='Test', color='black')
       
        ax.set_ylabel('$\%$ diff. to s.s.')
        ax.set_xlabel('Quarters')
        if hh == 'RA':
            ax.set_title('RANK')
        else:
            ax.set_title('HANK')
    ax.legend(loc="best", frameon=True)  

    fig.tight_layout()
   
    return fig   
 
def trad_nontrad_decomp(models, HHs=['HA', 'RA-IM'], T_max=21):
    model_HA = models['HA']
    scale = model_HA.par.scale
    resp_dict = {}
    for HH in HHs:
        resp_dict[HH] = {}
        resp_dict[HH]['T'] = {}
        resp_dict[HH]['NT'] = {}
        model_ = models[HH] 
        alphaT = model_.par.alphaT
        etaT = model_.par.etaT
        resp_dict[HH]['dC']   = (model_.path.C[0,:]/model_.ss.C-1)*scale*100
        resp_dict[HH]['T']['dC']   = (model_.path.CT[0,:]/model_.ss.CT-1)*scale*100
        resp_dict[HH]['NT']['dC']   = (model_.path.CNT[0,:]/model_.ss.CNT-1)*scale*100
        resp_dict[HH]['T']['dP'] = - etaT*(model_.path.PT[0,:]/model_.path.P[0,:] -1)*scale*100
        resp_dict[HH]['NT']['dP'] = - etaT*(model_.path.PNT[0,:]/model_.path.P[0,:] -1)*scale*100
        resp_dict[HH]['T']['test']  = resp_dict[HH]['dC'] + resp_dict[HH]['T']['dP']
        resp_dict[HH]['NT']['test'] = resp_dict[HH]['dC'] + resp_dict[HH]['NT']['dP']
        
    parvalslabels = ['Non-tradeable C', 'Tradeable C']
        
    #T_max = 21
    lwidth = 2.3
    testplot = False

    ncols,nrows = 2,2
    #set_palette("colorblind")
    fig = plt.figure(figsize=(4.3*ncols/1.1,3.6*nrows/1.2))

    temp = 0 
    for j,sec in enumerate(['NT', 'T']):    
        for i,HH in enumerate(['HA', 'RA-IM']):
            ax = fig.add_subplot(nrows,ncols,temp+1)
            ax.plot(np.zeros(T_max), '-', color='black')   
            #ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.plot(np.arange(T_max),resp_dict[HH][sec]['dC'][:T_max],label='Total', linewidth=lwidth, linestyle='-')
            ax.plot(np.arange(T_max),resp_dict[HH]['dC'][:T_max],label='Income effect', linewidth=lwidth, linestyle='--')
            ax.plot(np.arange(T_max),resp_dict[HH][sec]['dP'][:T_max],label='Substitution effect', linewidth=lwidth, linestyle=':')
                  
            ax.set_xlim([0, T_max-1])
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.set_xticks(np.arange(0, T_max, 4))
            if testplot:
                ax.plot(np.arange(T_max),resp_dict[HH][sec]['test'][:T_max], '--', label='Test', color='black')

            ax.set_ylabel('$\%$ diff. to s.s.')
            ax.set_xlabel('Quarters')
            if HH == 'RA-IM':
                ax.set_title('RANK')
            else:
                ax.set_title('HANK')
            if temp==0:
                ax.legend(loc="best", frameon=True)  
            temp +=1
            if i==0:
                pad=0.8
                ax.annotate(parvalslabels[j], xy=(-0.5, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size=18, ha='right', va='center', rotation=90)
    fig.tight_layout()
    return fig 

def compare_models(modellist, varlist, T_max, lwidth, model_labels, var_labels, abs_value=[], share_var=None):   
    scale=True
    ncols,nrows = len(modellist),1
    set_palette("colorblind")
    fig = plt.figure(figsize=(4.3*ncols,3.6*nrows))
    
    for i,model in enumerate(modellist):
        ax = fig.add_subplot(nrows,ncols,i+1)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.plot(np.zeros(T_max), '-', color='black')   

        if scale:
            scaleval = getattr(model.par,'scale')

        for j,var in enumerate(varlist):
            if var in abs_value:
                dVar = (getattr(model.path,var)[0,:] - getattr(model.ss,var))*scaleval
            elif share_var is not None:
                dVar = 100*((getattr(model.path,var)[0,:]-getattr(model.ss,var) )/ getattr(model.ss,share_var))*scaleval
            else:
                dVar = 100*(getattr(model.path,var)[0,:] / getattr(model.ss,var)-1)*scaleval
            ax.plot(np.arange(T_max),dVar[:T_max], label=var_labels[j], linewidth=lwidth)

        ax.set_xlim([0, T_max-1])

        ax.set_xlabel('Quarters')
        if ncols>1:
            ax.set_title(model_labels[i])


    ax.legend(loc="best", frameon=True)  
    fig.tight_layout()
   
    return fig   


def show_IRFs_new(models,paths,abs_value=[],pctp=[], T_max=None,labels=[None], pathlabels=None, shocktitle=None, share={}, 
                  do_sumplot=True, scale=False, ldash=None, colors=None, lwidth=1.3, customscale={}, palette=None, maxrow=4,
                   figsize=None, ratio_plot={}, lfsize=15, legend_window=0):
    

    model = models[0]
    legendbool = True
    par = model.par
    if T_max is None: T_max = par.T
    

    # figures
    varnames = paths 
    if pathlabels is None:
        pathlabels = paths
    else:
        pathlabels = pathlabels
        
    if do_sumplot:
        nfixpl = 3 
    else:
        nfixpl = 0   
    path_is_list = False 
    if isinstance(varnames[0],list):
        path_is_list = True
    if path_is_list:
        num = len(varnames[0])+nfixpl
    else:
        num = len(varnames)+nfixpl
    nrows = num//maxrow+1
    ncols = np.fmin(num,maxrow)
    if num%maxrow == 0: nrows -= 1 
    if figsize is None:
        fig = plt.figure(figsize=(4.3*ncols,3.6*nrows))
    else:
        fig = plt.figure(figsize=(figsize[0]*ncols,figsize[1]*nrows))

    for i in range(num):
        ax = fig.add_subplot(nrows,ncols,i+1)
        for j, model_ in enumerate(models):  
            if path_is_list:
                varnames = paths[j]

            if scale:
                scaleval = getattr(model_.par,'scale')
            else:
                scaleval = 1 
            if ldash is None:
                lstyle = '-' 
            else:
                lstyle=ldash[j]

            if colors is None:               
                col = None
            elif palette is not None:
                temp = 1
                col = color_palette(palette, temp+len(models))[temp+j]
                if colors is not None:
                    col = colors[j]
            else:
                col = colors[j]

            label = labels[j]
            
            if do_sumplot:    
                k = 1
                if i==0: # National accounts 
                    dY = utils.get_dX('Y', model_, False, scaleval)
                    dCH = utils.get_dX('CH', model_, True, scaleval)   /  getattr(model.ss,'Y')
                    dCNT = utils.get_dX('CNT', model_, True, scaleval)   /  getattr(model.ss,'Y')
                    dNX = utils.get_dX('NX', model_, True, scaleval) /  getattr(model.ss,'Y')
                    ax.plot(np.arange(T_max),dY[:T_max]*100,label='Y', linestyle=lstyle)
                    ax.plot(np.arange(T_max),dCH[:T_max]*100,label='CH', linestyle=lstyle)
                    ax.plot(np.arange(T_max),dCNT[:T_max]*100,label='CNT', linestyle=lstyle)
                    ax.plot(np.arange(T_max),dNX[:T_max]*100,label='NX', linestyle=lstyle)
                    ax.plot(np.zeros(T_max), '-', color='black')
                    ax.set_ylabel('% diff. to of s.s. in units of Y')                           
                    ax.legend(loc="best", frameon=True)  
                    ax.set_title('National Accounts')
                
                elif i==1: # consumption 
                    dCH = utils.get_dX('CH', model_, False, scaleval)  
                    dCF = utils.get_dX('CF', model_, False, scaleval)  
                    dCH_s = utils.get_dX('CH_s', model_, False, scaleval)  
                    ax.plot(np.arange(T_max),dCH[:T_max]*100,label='CH', linestyle=lstyle)
                    ax.plot(np.arange(T_max),dCF[:T_max]*100,label='CF', linestyle=lstyle)
                    ax.plot(np.arange(T_max),dCH_s[:T_max]*100,label='CH_s', linestyle=lstyle)
                    ax.plot(np.zeros(T_max), '-', color='black')
                    ax.set_ylabel('% diff. to s.s.')                           
                    ax.legend(loc="best", frameon=True)  
                    ax.set_title('Consumption')
                elif i==2: # prices
                    dP = utils.get_dX('P', model_, False, scaleval)  
                    dPH = utils.get_dX('PH', model_, False, scaleval) 
                    dPNT = utils.get_dX('PNT', model_, False, scaleval)  
                    dPF = utils.get_dX('PF', model_, False, scaleval)  
                    dE = utils.get_dX('E', model_, False, scaleval)  
                    dQ = utils.get_dX('Q', model_, False, scaleval)  
                    ax.plot(np.zeros(T_max), '-', color='black')
                    ax.plot(np.arange(T_max),dP[:T_max]*100,label='P', linestyle=lstyle)
                    ax.plot(np.arange(T_max),dPH[:T_max]*100,label='PH', linestyle=lstyle)
                    ax.plot(np.arange(T_max),dPNT[:T_max]*100,label='PNT', linestyle=lstyle)
                    ax.plot(np.arange(T_max),dPF[:T_max]*100,label='PF', linestyle=lstyle)
                    ax.plot(np.arange(T_max),dE[:T_max]*100,label='E', linestyle=lstyle)
                    ax.plot(np.arange(T_max),dQ[:T_max]*100,label='Q', linestyle=lstyle)
                    ax.set_ylabel('% diff. to of s.s.')                           
                    ax.legend(loc="best", frameon=True)   
                    ax.set_title('Prices')
                    

            if i >= nfixpl:               
                varname = varnames[i-nfixpl]
                ax.set_title(varname)     
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                ax.set_xticks(np.arange(0, T_max, 4))

                if varname != 'TradeBalance' and varname not in ratio_plot:
                    pathvalue = getattr(model_.path,varname)[0,:]      
                    ssvalue = getattr(model_.ss,varname)
    
                if varname in abs_value:                     
                    ax.plot(np.arange(T_max),(pathvalue[:T_max]-ssvalue)*scaleval,label=label, linestyle=lstyle, color=col,linewidth=lwidth)
                    ax.set_ylabel('abs. diff. to s.s.')
                    ax.plot(np.zeros(T_max), '-', color='black')
                    ax.set_title(pathlabels[i-nfixpl])
                elif varname in pctp:
                    ax.plot(np.arange(T_max),100*(pathvalue[:T_max]-ssvalue)*scaleval,label=label, linestyle=lstyle, color=col,linewidth=lwidth)
                    ax.set_ylabel('\%-points diff. to s.s.')
                    ax.plot(np.zeros(T_max), '-', color='black')
                    ax.set_title(pathlabels[i-nfixpl])
                elif varname in share:
                    dpathvalue = 100*(pathvalue[:T_max]-ssvalue)*scaleval / getattr(model_.ss,share[varname])  
                    ax.plot(np.arange(T_max),dpathvalue,label=label, linestyle=lstyle, color=col,linewidth=lwidth)
                    #ax.set_ylabel('Pct. points diff. to s.s.')
                    ax.set_ylabel(f'$%$ diff. to s.s. in units of {share[varname]}')
                    ax.plot(np.zeros(T_max), '-', color='black')
                    ax.set_title(pathlabels[i-nfixpl])
                elif varname =='TradeBalance':            
                    pathvalue_IM = getattr(model_.path,'Imports')[0,:]    
                    pathvalue_EX = getattr(model_.path,'Exports')[0,:]   
                    ssvalue_IM = ssvalue = getattr(model_.ss,'Imports')
                    ssvalue_EX = ssvalue = getattr(model_.ss,'Exports')
                    dIM = 100*(pathvalue_IM[:T_max]-ssvalue_IM)*scaleval / ssvalue_IM  
                    dEX = 100*(pathvalue_EX[:T_max]-ssvalue_EX)*scaleval / ssvalue_EX      
                    dNX = dEX -  dIM                    
                    ax.plot(np.arange(T_max),dEX,label='Exports', linestyle=lstyle, color=col,linewidth=lwidth)
                    ax.plot(np.arange(T_max),dIM,label='Imports', linestyle='--', color=col,linewidth=lwidth)
                    ax.plot(np.arange(T_max),dNX,label='NX', linestyle=':', color=col,linewidth=lwidth)
                    
                    #ax.set_ylabel('Pct. points diff. to s.s.')
                    ax.set_ylabel('% diff. to s.s.')
                    ax.plot(np.zeros(T_max), '-', color='black')
                    ax.set_title('Exports/imports')
                    if legendbool:
                        ax.legend(frameon=True, prop={'size': 12})
                    legendbool = False 
                elif varname == 'Walras':
                    dWalras = pathvalue #* scaleval / model_.path.Y[0,:]  
                    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
                    ax.plot(np.arange(T_max),scaleval*dWalras[:T_max]*100,label=label, linestyle=lstyle, color=col,linewidth=lwidth)
                    ax.set_ylabel('% diff. to s.s. in units of Y')
                    ax.plot(np.zeros(T_max), '-', color='black')
                    ax.set_title(pathlabels[i-nfixpl])
                elif varname  in customscale:
                    dpathvalue = (pathvalue[:T_max]-ssvalue)*scaleval / (customscale[varname]*scaleval)
                    ax.plot(np.arange(T_max),dpathvalue,label=label, linestyle=lstyle, color=col,linewidth=lwidth)
                    if 'descr' in customscale:
                        ax.set_ylabel(customscale['descr'])
                    else:
                        ax.set_ylabel(f'% diff. to s.s. scaled')
                    ax.plot(np.zeros(T_max), '-', color='black')
                    ax.set_title(pathlabels[i-nfixpl])                   
                elif varname  in ratio_plot:
                    dNum = (getattr(model_.path,ratio_plot[varname][0])[0,:]/getattr(model_.ss,ratio_plot[varname][0])-1)*100*scaleval
                    dDenom = (getattr(model_.path,ratio_plot[varname][1])[0,:]/getattr(model_.ss,ratio_plot[varname][1])-1)*100*scaleval
                    ax.plot(np.arange(T_max),dNum[:T_max]-dDenom[:T_max],label=label, linestyle=lstyle, color=col,linewidth=lwidth)
                    if 'descr' in ratio_plot:
                        ax.set_ylabel(ratio_plot['descr'])
                    else:
                        ax.set_ylabel('\% diff. to s.s.')
                    ax.plot(np.zeros(T_max), '-', color='black')
                    ax.set_title(pathlabels[i-nfixpl])     
                elif varname == 'NX':   
                    pathvalue_IM = getattr(model_.path,'Imports')[0,:]    
                    pathvalue_EX = getattr(model_.path,'Exports')[0,:]   
                    ssvalue_IM = getattr(model_.ss,'Imports')
                    ssvalue_EX = getattr(model_.ss,'Exports')
                    dIM = 100*(pathvalue_IM[:T_max]-ssvalue_IM)*scaleval / ssvalue_IM  
                    dEX = 100*(pathvalue_EX[:T_max]-ssvalue_EX)*scaleval / ssvalue_EX      
                    dNX = dEX -  dIM  
                    ax.plot(np.arange(T_max),dNX,label=label, linestyle=lstyle, color=col,linewidth=lwidth)
                    ax.set_ylabel('\% diff. to s.s.')
                    ax.plot(np.zeros(T_max), '-', color='black')                                              
                else:
                    if abs(ssvalue) > 0: 
                        ax.plot(np.arange(T_max),((pathvalue[:T_max]-ssvalue)*scaleval/ssvalue)*100,label=label, linestyle=lstyle, color=col,linewidth=lwidth)
                        ax.set_ylabel('$\%$ diff. to s.s.')
                        ax.plot(np.zeros(T_max), '-', color='black')
                        ax.set_title(pathlabels[i-nfixpl])
                    else:
                        ax.plot(np.arange(T_max),((pathvalue[:T_max]-ssvalue)*scaleval)*100,label=label, linestyle=lstyle, color=col,linewidth=lwidth)
                        ax.set_ylabel('$\%$ diff. to s.s.')
                        ax.plot(np.zeros(T_max), '-', color='black')
                        ax.set_title(pathlabels[i-nfixpl])                           
                #else:
                ax.set_xlim([0, T_max-1])
                #    ax.plot(np.arange(T_max),pathvalue[:T_max],label=label, linestyle=lstyle)
                if i>= ncols*(nrows-1):
                    ax.set_xlabel('Quarters', fontsize=16)  
                if len(models) > 1 and i == legend_window: ax.legend(frameon=True, prop={'size': lfsize})
    if shocktitle is not None:
        fig.suptitle(shocktitle, fontsize=16)
        
    fig.tight_layout(pad=1.6)
    plt.show()
    print('')
    return fig

  
def show_IRFs_new_robust(models,paths,parname, parvals,abs_value=[],pctp=[], T_max=None,labels=[None], pathlabels=None, shocktitle=None, share={}, parvalslabels=None,
                  do_sumplot=True, scale=False, ldash=None, colors=None, lwidth=1.3, customscale={}, palette=None, maxrow=4, figsize=None, ratio_plot={}):
    
  
    key = list(models[parvals[0]].keys())[0]
    model = models[parvals[0]][key]
    legendbool = True
    par = model.par
    if T_max is None: T_max = par.T
    
    N_parvals = len(parvals)
    
    # full_list
    full_list = []
    full_list.append(('paths',paths))
    
    # figures
    varnames = paths 
    if pathlabels is None:
        pathlabels = paths
    else:
        pathlabels = pathlabels
        
    if do_sumplot:
        nfixpl = 3 
    else:
        nfixpl = 0   
    
    #num = len(varnames)+nfixpl
    nrows = N_parvals
    ncols = len(paths)
    num = nrows*ncols

    #if num%maxrow == 0: nrows -= 1 
    if figsize is None:
        fig = plt.figure(figsize=(4.3*ncols,3.6*nrows))
    else:
        fig = plt.figure(figsize=(figsize[0]*ncols,figsize[1]*nrows))


    for i,ipar in enumerate(parvals):

        for jpaths in range(ncols):
            ax = fig.add_subplot(nrows,ncols,jpaths+i*ncols+1)


            if i==N_parvals-1:
                ax.set_xlabel('Quarters')
            if i==0:
                ax.set_title(pathlabels[jpaths])   

            
            if jpaths==0:
                pad=0.8
                parlabel = ipar if parvalslabels is None else parvalslabels[i]
                ax.annotate(parlabel, xy=(-0.5, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size=18, ha='right', va='center', rotation=90)
            for j, hhkey in enumerate(models[ipar]):  
                model_ = models[ipar][hhkey]
                if scale:
                    scaleval = getattr(model_.par,'scale')
                else:
                    scaleval = 1 
                if ldash is None:
                    lstyle = '-' 
                else:
                    lstyle=ldash[j]

                if colors is None:               
                    col = None
                elif palette is not None:
                    temp = 1
                    col = color_palette(palette, temp+len(models))[temp+j]
                else:
                    col = colors[j]

                label = labels[j]

                if i >= nfixpl:               
                    varname = varnames[jpaths]
                    #ax.set_title(varname)     
                    
                    if varname != 'TradeBalance' and varname not in ratio_plot:
                        pathvalue = getattr(model_.path,varname)[0,:]      
            
                        #if not np.isnan(getattr(model_.ss,varname)):   
                        ssvalue = getattr(model_.ss,varname)
        
                    if varname in abs_value:                     
                        ax.plot(np.arange(T_max),(pathvalue[:T_max]-ssvalue)*scaleval,label=label, linestyle=lstyle, color=col,linewidth=lwidth)
                        ax.set_ylabel('abs. diff. to s.s.')
                        ax.plot(np.zeros(T_max), '-', color='black')
                    elif varname in pctp:
                        ax.plot(np.arange(T_max),100*(pathvalue[:T_max]-ssvalue)*scaleval,label=label, linestyle=lstyle, color=col,linewidth=lwidth)
                        ax.set_ylabel('Pct. points diff. to s.s.')
                        ax.plot(np.zeros(T_max), '-', color='black')
                        #ax.set_title(pathlabels[i-nfixpl])
                    elif varname in share:
                        dpathvalue = 100*(pathvalue[:T_max]-ssvalue)*scaleval / getattr(model_.ss,share[varname])  
                        ax.plot(np.arange(T_max),dpathvalue,label=label, linestyle=lstyle, color=col,linewidth=lwidth)
                        #ax.set_ylabel('Pct. points diff. to s.s.')
                        ax.set_ylabel(f'% diff. to s.s. in units of {share[varname]}')
                        ax.plot(np.zeros(T_max), '-', color='black')
                        #ax.set_title(pathlabels[i-nfixpl])
                    elif varname =='TradeBalance':            
                        pathvalue_IM = getattr(model_.path,'Imports')[0,:]    
                        pathvalue_EX = getattr(model_.path,'Exports')[0,:]   
                        ssvalue_IM = ssvalue = getattr(model_.ss,'Imports')
                        ssvalue_EX = ssvalue = getattr(model_.ss,'Exports')
                        dIM = 100*(pathvalue_IM[:T_max]-ssvalue_IM)*scaleval / ssvalue_IM  
                        dEX = 100*(pathvalue_EX[:T_max]-ssvalue_EX)*scaleval / ssvalue_EX      
                        dNX = dEX -  dIM                    
                        ax.plot(np.arange(T_max),dEX,label='Exports', linestyle=lstyle, color=col,linewidth=lwidth)
                        ax.plot(np.arange(T_max),dIM,label='Imports', linestyle='--', color=col,linewidth=lwidth)
                        ax.plot(np.arange(T_max),dNX,label='NX', linestyle=':', color=col,linewidth=lwidth)
                        
                        #ax.set_ylabel('Pct. points diff. to s.s.')
                        ax.set_ylabel('% diff. to s.s.')
                        ax.plot(np.zeros(T_max), '-', color='black')
                        #ax.set_title('Exports/imports')
                        if legendbool:
                            ax.legend(frameon=True, prop={'size': 12})
                        legendbool = False 
                    elif varname == 'Walras':
                        dWalras = pathvalue #* scaleval / model_.path.Y[0,:]  
                        ax.plot(np.arange(T_max),dWalras[:T_max]*100,label=label, linestyle=lstyle, color=col,linewidth=lwidth)
                        ax.set_ylabel('% diff. to s.s. in units of Y')
                        ax.plot(np.zeros(T_max), '-', color='black')
                        #ax.set_title(pathlabels[i-nfixpl])
                    elif varname  in customscale:
                        dpathvalue = (pathvalue[:T_max]-ssvalue)*scaleval / (customscale[varname]*scaleval)
                        ax.plot(np.arange(T_max),dpathvalue,label=label, linestyle=lstyle, color=col,linewidth=lwidth)
                        #ax.set_ylabel('Pct. points diff. to s.s.')
                        #ax.set_ylabel(f'% diff. to s.s. scaled')
                        if 'descr' in customscale:
                            ax.set_ylabel(customscale['descr'])
                        else:
                            ax.set_ylabel(f'% diff. to s.s. scaled')
                        ax.plot(np.zeros(T_max), '-', color='black')
                        #ax.set_title(pathlabels[i-nfixpl])                   
                    elif varname  in ratio_plot:
                        dNum = (getattr(model_.path,ratio_plot[varname][0])[0,:]/getattr(model_.ss,ratio_plot[varname][0])-1)*100*scaleval
                        dDenom = (getattr(model_.path,ratio_plot[varname][1])[0,:]/getattr(model_.ss,ratio_plot[varname][1])-1)*100*scaleval
                        ax.plot(np.arange(T_max),dNum[:T_max]-dDenom[:T_max],label=label, linestyle=lstyle, color=col,linewidth=lwidth)
                        #ax.set_ylabel('Pct. points diff. to s.s.')
                        #ax.set_ylabel(f'% diff. to s.s. scaled')
                        if 'descr' in ratio_plot:
                            ax.set_ylabel(ratio_plot['descr'])
                        else:
                            ax.set_ylabel(f'% diff. to s.s. scaled')
                        ax.plot(np.zeros(T_max), '-', color='black')
                        #ax.set_title(pathlabels[i-nfixpl])   
                    elif varname == 'NX':   
                        pathvalue_IM = getattr(model_.path,'Imports')[0,:]    
                        pathvalue_EX = getattr(model_.path,'Exports')[0,:]   
                        ssvalue_IM = getattr(model_.ss,'Imports')
                        ssvalue_EX = getattr(model_.ss,'Exports')
                        dIM = 100*(pathvalue_IM[:T_max]-ssvalue_IM)*scaleval / ssvalue_IM  
                        dEX = 100*(pathvalue_EX[:T_max]-ssvalue_EX)*scaleval / ssvalue_EX      
                        dNX = dEX -  dIM  
                        ax.plot(np.arange(T_max),dNX,label=label, linestyle=lstyle, color=col,linewidth=lwidth)
                        ax.set_ylabel('% diff. to s.s.')
                        ax.plot(np.zeros(T_max), '-', color='black')                          
                    else:
                        if abs(ssvalue) > 0: 
                            ax.plot(np.arange(T_max),((pathvalue[:T_max]-ssvalue)*scaleval/ssvalue)*100,label=label, linestyle=lstyle, color=col,linewidth=lwidth)
                            ax.set_ylabel('% diff. to s.s.')
                            ax.plot(np.zeros(T_max), '-', color='black')
                            #ax.set_title(pathlabels[i-nfixpl])
                        else:
                            ax.plot(np.arange(T_max),((pathvalue[:T_max]-ssvalue)*scaleval)*100,label=label, linestyle=lstyle, color=col,linewidth=lwidth)
                            ax.set_ylabel('% diff. to s.s.')
                            ax.plot(np.zeros(T_max), '-', color='black')
                            #ax.set_title(pathlabels[i-nfixpl])                           
                    #else:
                    #ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                    #    ax.plot(np.arange(T_max),pathvalue[:T_max],label=label, linestyle=lstyle)
                    if i>= ncols*(nrows-1):
                        ax.set_xlabel('Quarters', fontsize=16)  
                    if len(models) > 1 and i == 0 and jpaths==0: ax.legend(frameon=True, prop={'size': 15})
    if shocktitle is not None:
        fig.suptitle(shocktitle, fontsize=16)
        
    
    fig.tight_layout(pad=1.6)
    plt.show()
    print('')
    return fig


def show_IRFs_new_robust_v2(models,paths,parname, parvals,abs_value=[],pctp=[], T_max=None,labels=[None], pathlabels=None, shocktitle=None, share={}, parvalslabels=None,
                  do_sumplot=True, scale=False, ldash=None, colors=None, lwidth=1.3, customscale={}, palette=None, maxrow=4, figsize=None, ratio_plot={}, do_reduction=False):
    
  
    key = list(models[parvals[0]].keys())[0]
    model = models[parvals[0]][key]
    legendbool = True
    par = model.par
    if T_max is None: T_max = par.T
    
    N_parvals = len(parvals)
    
    # full_list
    full_list = []
    full_list.append(('paths',paths))
    
    # figures
    varnames = paths 
    if pathlabels is None:
        pathlabels = paths
    else:
        pathlabels = pathlabels
        
    if do_sumplot:
        nfixpl = 3 
    else:
        nfixpl = 0   
    
    #num = len(varnames)+nfixpl
    nrows = N_parvals
    ncols = len(paths)
    num = nrows*ncols

    #if num%maxrow == 0: nrows -= 1 
    if figsize is None:
        fig = plt.figure(figsize=(4.3*ncols,3.6*nrows))
    else:
        fig = plt.figure(figsize=(figsize[0]*ncols,figsize[1]*nrows))


    for i,ipar in enumerate(parvals):

        for jpaths in range(ncols):
            ax = fig.add_subplot(nrows,ncols,jpaths+i*ncols+1)
            ax.set_xticks(np.arange(0, T_max, 4))

            if i==N_parvals-1:
                ax.set_xlabel('Quarters')
            if i==0:
                ax.set_title(pathlabels[jpaths])   

            
            if jpaths==0:
                pad=0.8
                parlabel = ipar if parvalslabels is None else parvalslabels[i]
                ax.annotate(parlabel, xy=(-0.5, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size=18, ha='right', va='center', rotation=90)
            for j, hhkey in enumerate(models[ipar]):  
                model_ = models[ipar][hhkey]
                if scale:
                    scaleval = getattr(model_.par,'scale')
                else:
                    scaleval = 1 
                if ldash is None:
                    lstyle = '-' 
                else:
                    lstyle=ldash[j]

                if colors is None:               
                    col = None
                elif palette is not None:
                    temp = 1
                    col = color_palette(palette, temp+len(models))[temp+j]
                else:
                    col = colors[j]

                label = labels[j]

                if i >= nfixpl:               
                    varname = varnames[jpaths]
                    #ax.set_title(varname)     

                    if do_reduction:
                        pathvalue = getattr(model_.path,varname)                 
                    else:
                        pathvalue = getattr(model_.path,varname)[0,:]      
                    ssvalue = getattr(model_.ss,varname)
                            
                    if varname in abs_value:                     
                        ax.plot(np.arange(T_max),(pathvalue[:T_max]-ssvalue)*scaleval,label=label, linestyle=lstyle, color=col,linewidth=lwidth)
                        ax.set_ylabel('abs. diff. to s.s.')
                        ax.plot(np.zeros(T_max), '-', color='black')
                    elif varname in pctp:
                        ax.plot(np.arange(T_max),100*(pathvalue[:T_max]-ssvalue)*scaleval,label=label, linestyle=lstyle, color=col,linewidth=lwidth)
                        ax.set_ylabel('Pct. points diff. to s.s.')
                        ax.plot(np.zeros(T_max), '-', color='black')
                        #ax.set_title(pathlabels[i-nfixpl])                                        
                    else:
                        if abs(ssvalue) > 0: 
                            ax.plot(np.arange(T_max),((pathvalue[:T_max]-ssvalue)*scaleval/ssvalue)*100,label=label, linestyle=lstyle, color=col,linewidth=lwidth)
                            ax.set_ylabel('$\%$ diff. to s.s.')
                            ax.plot(np.zeros(T_max), '-', color='black')
                            #ax.set_title(pathlabels[i-nfixpl])
                        else:
                            ax.plot(np.arange(T_max),((pathvalue[:T_max]-ssvalue)*scaleval)*100,label=label, linestyle=lstyle, color=col,linewidth=lwidth)
                            ax.set_ylabel('$\%$ diff. to s.s.')
                            ax.plot(np.zeros(T_max), '-', color='black')
                            #ax.set_title(pathlabels[i-nfixpl])                           
                    #else:
                    #ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                    #    ax.plot(np.arange(T_max),pathvalue[:T_max],label=label, linestyle=lstyle)
                    if i>= ncols*(nrows-1):
                        ax.set_xlabel('Quarters', fontsize=16)  
                    if len(models) > 1 and i == 0 and jpaths==0: ax.legend(frameon=True, prop={'size': 15})
    if shocktitle is not None:
        fig.suptitle(shocktitle, fontsize=16)
        
    
    fig.tight_layout(pad=1.6)
    plt.show()
    print('')
    return fig


def PE_MonPol_shock(model, lin=False, do_KMV=False):
    
    
    #x_KMV =   np.array([0.241,0.746,1.096,1.711,2.675,3.904,5.175,7.325,9.342,12.5,14.474,17.061,18.575] )
    #y_KMV = np.array([0.05,0.03,0.009,-0.006,-0.015,-0.02,-0.02,-0.02,-0.018,-0.018,-0.017,-0.013,-0.012])

    # shock
    x_KMV_shock = np.array([0.02, 0.3,0.8,1.2,2.0,2.4,3.2,3.9,4.7,5.1,6.1,6.6,7.6,8.5,9.2, 10, 11, 12 , 13])
    y_KMV_shock = np.array([ -0.666,-0.582,-0.453,-0.360,-0.252,-0.214,-0.145,-0.107,-0.072,-0.055,-0.035,-0.036,-0.024,-0.019,-0.020, 0, 0, 0, 0])/4 # convert to quarterly 
    cs_shock = CubicSpline(x_KMV_shock, y_KMV_shock)
    y_shock_int = cs_shock(np.arange(0,10))
    KMV_shock = np.zeros(model.par.T)
    KMV_shock[:10] = y_shock_int

    #x_KMV =  np.array([0.02,0.21,0.39,0.55,0.68,0.83,1.33,1.56,1.74,2.06,2.46,2.65,2.90,3.78,3.94,4.29,5.21,5.48,5.73,6.61,7.04,8.04,8.45,9.47,9.84,10.8,11.3,12.3,12.7,13.7,14.1,15.1,15.6,16.5,16.9,17.9,18.3])
    #y_KMV = np.array([1.402657,1.175885,1.001021,0.858584,0.761518,0.670988,0.412557,0.335228,0.277252,0.174203,
    #                    0.090806,0.045861,0.007537,-0.09406,-0.10662,-0.13170,-0.17477,-0.18054,-0.19289,
    #                    -0.18410,-0.18949,-0.18689,0.185828,-0.17020,-0.16272, -0.1600,  -0.152, -0.1301,-0.14211,-0.11358,
    #                    -0.10615,-0.10996,-0.08929,-0.0933,-0.09225,-0.07662, -0.0756])  /10

    x_KMV =   np.array([0, 1, 2,	3,	4,	5,	6,	7,	8,	9,	10,	11,	12,	13,	14,	15	,16,	17,	18,	19, 20] ) 
    y_KMV = np.array([0.14*1000, 62,	19,	3	,-8	,-14,	-18,	-19,	-19,	-18,	-16,	-13,	-13,-13	,-12	,-12,	-12,	-12,	-11,	-11, -11])/1000
    
    cs = CubicSpline(x_KMV, y_KMV)
    y_int = cs(np.arange(0,21))

    
    # old shock
    impact = -0.28 / (np.sum(- 0.61**np.arange(4)) * 100)    
    dra_old = - impact* 0.61**np.arange(model.par.T)    
    
    # new KMV shock 
    dra = KMV_shock / 100 

    beta = 1/(1+model.ss.ra)
    MPC_R = (1-beta)*beta**np.arange(model.par.T)
    CR = np.zeros(model.par.T)
    A_test = np.zeros(model.par.T)
    I = model.ss.A + model.ss.C - (model.ss.A*(1+model.ss.ra))
    dC_inf  = 0
    C_ss = np.sum(model.ss.c*model.ss.D)
    for t in range(model.par.T):
        dC_inf += model.par.CRRA * (1-beta) * beta**t * np.sum(dra[t+1:]/(1+model.ss.ra))
        
    reval = model.ss.A * (1-beta)
    for tt in range(model.par.T):
        t = model.par.T-1-tt
        if t == model.par.T-1:
            CR[t] = model.ss.C            
        else:
            CR[t] =  model.ss.C  - 1/model.par.CRRA * beta * np.sum(dra[t+1:]) + reval * dra[0]
            
    for t in range(model.par.T):
        if t >0:
            A_test[t] = I + A_test[t-1]*(1+model.ss.ra + dra[t]) - CR[t]
        else:
            A_test[t] = I + model.ss.A*(1+model.ss.ra + dra[t]) - CR[t]
            
    if lin:            
        model.compute_jac_hh(do_print=False)   
        dC = model.jac_hh.C_ra @ dra  
    else:
        model._set_inputs_hh_all_ss()
        model.path.ra[0,:] = model.ss.ra + dra
        model.solve_hh_path()
        model.simulate_hh_path()    
        
        dC = np.zeros(model.par.T)*np.nan
        for t in range(model.par.T):
            dC[t] = deepcopy(np.sum((model.path.c[t] * model.path.D[t])) - C_ss) 
        model._set_inputs_hh_all_ss()

    t = 21
    xaxis = np.arange(0,t)
    lsize = 1.8 
    set_palette("colorblind")
    fig = plt.figure(figsize=(4.7*1,3.2*1))
    ax = fig.add_subplot(1,1,1)
    ax.set_xticks(np.arange(0, t, 4))
    ax.plot(xaxis,np.zeros(t), '--', color='black')
    ax.plot(xaxis,(dC[:t]/model.ss.C)*100, label='HANK', color='C2', linewidth=lsize)
    ax.plot(xaxis,(CR[:t]/model.ss.C-1)*100, '-', label='RANK', color='C1', linewidth=lsize)
    if do_KMV:
        ax.plot(xaxis,y_int, '--', label='Kaplan et al. (2018)', color='C0', linewidth=lsize)
    ax.legend(frameon=True, prop={'size': 12})
    ax.set_xlabel('Quarters', fontsize=16)
    ax.set_ylabel('\% diff. to s.s.', fontsize=12)
    ax.set_xlim([0, t-1])
    fig.tight_layout()        
    
    return fig


#from seaborn import color_palette, set_palette

def plot_jac_columns(model_RA, model_HA, calc_jac=True):
    
    if calc_jac:
        model_HA._compute_jac_hh()
    
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


    columns = [0,4, 8, 12, 16, 20, 24]
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
    ax.set_xlim([0, Nquarters-1])
    ax.set_xlabel('Quarters', fontsize=16)
    # plt.ylabel('Damped oscillation')
    #ax.legend(frameon=True, fontsize=10)
    ax.set_ylabel(f'$dC$', fontsize=12)
    ax.set_xticks(np.arange(0, Nquarters, 4))


    ax = fig.add_subplot(1,2,2)
    ax.set_title(r'$\mathbf{M^r}$')

    l1_c,l2_c = l1, l2
    for i,col in enumerate(columns):   
        ax.plot(HA_dC_dr[:Nquarters, col], '-', label=l1_c, color='C2', linewidth=lsize, alpha=alphalist[i])
        ax.plot(RA_dC_dr[:Nquarters, col], '-', label=l2_c, color='C1', linewidth=lsize, alpha=alphalist[i])    
        l1_c, l2_c = '_nolegend_', '_nolegend_' # suppress extra legends

    ax.plot(np.zeros(Nquarters), '--', color='black')
    ax.set_xticks(np.arange(0, Nquarters, 4))
    ax.set_xlabel('Quarters', fontsize=16)
    # plt.ylabel('Damped oscillation')
    ax.legend(frameon=True, fontsize=12)
    ax.set_ylabel(f'$dC$', fontsize=12)
    ax.set_xlim([0, Nquarters-1])

    plt.tight_layout()
    fig.savefig(f'plots\calibration\M_columns.pdf')

    return fig 



def PE_MonPol_shock_Holm(model, lin=True):
    
    
    x_HPT = np.array([0,1*4,2*4,3*4, 4*4, 5*4])
    y_HPT = -np.array([ -0.35396039603960383,-0.40841584158415833,0.004950495049505177,-0.04455445544554426, -0.04455445544554426, 0.0])
    cs = CubicSpline(x_HPT, y_HPT)
    y_int = cs(np.arange(0,21))
    
    # now scale such that impact is 1 pct. decrease in first year     
    scale = 0.01 / np.sum( 0.42**np.arange(4))
    dra = - scale * 0.4**np.arange(model.par.T)
    
    plt.plot(x_HPT, y_HPT)
    plt.show()
    
    beta = 1/(1+model.ss.ra)
    MPC_R = (1-beta)*beta**np.arange(model.par.T)
    CR = np.zeros(model.par.T)
    A_test = np.zeros(model.par.T)
    I = model.ss.A + model.ss.C - (model.ss.A*(1+model.ss.ra))
    dC_inf  = 0
    for t in range(model.par.T):
        dC_inf += model.par.CRRA * (1-beta) * beta**t * np.sum(dra[t+1:]/(1+model.ss.ra))
        
    reval = model.ss.A * (1-beta)
    for tt in range(model.par.T):
        t = model.par.T-1-tt
        if t == model.par.T-1:
            CR[t] = model.ss.C            
        else:
            CR[t] =  model.ss.C  - 1/model.par.CRRA * beta * np.sum(dra[t+1:]) + reval * dra[0]
            
    for t in range(model.par.T):
        if t >0:
            A_test[t] = I + A_test[t-1]*(1+model.ss.ra + dra[t]) - CR[t]
        else:
            A_test[t] = I + model.ss.A*(1+model.ss.ra + dra[t]) - CR[t]
            
    if lin:            
        model.compute_jac_hh(do_print=False)   
        dC = model.jac_hh.C_ra @ dra  
    else:
        model._set_inputs_hh_ss()
        model.path.ra[0,:] = model.ss.ra + dra
        model.solve_hh_path()
        #prepare_simulation_1d_1d(model.par,model.sol,model.path.a,model.par.a_grid)
        model.simulate_hh_path()    
        
        dC = np.zeros(model.par.T)*np.nan
        for t in range(model.par.T):
            dC[t] = deepcopy(np.sum((model.path.c[t] * model.path.D[t])) - model.ss.C) 
            
    t = 21
    # t=600
    xaxis = np.arange(0,t)
    lsize = 1.8 
    set_palette("colorblind")
    fig = plt.figure(figsize=(4.7*1,3.2*1))
    ax = fig.add_subplot(1,1,1)
    ax.plot(xaxis,np.zeros(t), '--', color='black')
    ax.plot(xaxis,(dC[:t]/model.ss.C)*100, label='HANK', color='C2', linewidth=lsize)
    ax.plot(xaxis,(CR[:t]/model.ss.C-1)*100, '-', label='Ricardian', color='C1', linewidth=lsize)
    ax.plot(xaxis,y_int, '--', label='Kaplan et al. (2018)', color='C0', linewidth=lsize)
    ax.legend(frameon=True, prop={'size': 10})
    ax.set_xlabel('Quarters', fontsize=12)
    ax.set_ylabel('% diff. to s.s.', fontsize=12)
    fig.tight_layout() 
    plt.show()
       
    # yearly
    t = 21
    dC_HA_ann = [sum(dC[j*4:(1+j)*4]) for j in range(6)] / (model.ss.C*4) * 100
    dC_RA_ann = ([sum(CR[j*4:(1+j)*4]) for j in range(6)] / (model.ss.C*4)-1) * 100
    
    
    xaxis = np.arange(0,6)
    lsize = 1.8 
    set_palette("colorblind")
    fig = plt.figure(figsize=(4.7*1,3.2*1))
    ax = fig.add_subplot(1,1,1)
    ax.plot(xaxis,np.zeros(6), '--', color='black')
    ax.plot(xaxis,dC_HA_ann, label='HANK', color='C2', linewidth=lsize)
    ax.plot(xaxis,dC_RA_ann, '-', label='Ricardian', color='C1', linewidth=lsize)
    ax.plot(xaxis,y_HPT, '--', label='Kaplan et al. (2018)', color='C0', linewidth=lsize)
    ax.legend(frameon=True, prop={'size': 10})
    ax.set_xlabel('Quarters', fontsize=12)
    ax.set_ylabel('% diff. to s.s.', fontsize=12)
    fig.tight_layout() 
    plt.show()
    
    
    return fig



def vary_irfs(shock,varlist,scalevar,paramlist,paramvals,HH_type='HA',jump=0.001,T_max=40, boolpars=False, varlistlabels=None, ncols=3,
            foreignshock=None,scalesize=0.01, upars=None, absval=True, alphacolor=True, modellabels=None, pctp=[], upars_foreign={},
            reuse_HH_jac = False, homotopic_cont = False, reset_init_vals=True, model_base=None, title=None):
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
    legendplacement = 0
    N_models = len(paramvals)
    if isinstance(varlist,list):
        N_vars = len(varlist)
    else:
        N_vars = 1
        varlist = [varlist]
    #models = [HANKModelClass() for i in range(N_models)]
    # dXs = np.zeros((N_models,N_vars,models[0].par.T))
    
    if foreignshock:
        model_foreign = GetForeignEcon.get_foreign_econ(shocksize=0.001, upars=upars_foreign)

    if isinstance(paramlist, list):
        pass
    else:
        paramlist = [paramlist]
    N_params = len(paramlist)

    paramlist_dict = [np.nan] * N_models

    compiled = False 
    model_base_bool = True
    if model_base is None: model_base_bool = False
    for i in range(N_models):
        if not compiled and model_base_bool == False:
            model_ = HANKModelClass()
            init_vals = model_.par.x0.copy()  
            dXs = np.zeros((N_models,N_vars,model_.par.T))
        elif not compiled:
            model_ = model_base.copy()
            dXs = np.zeros((N_models,N_vars,model_.par.T))            
            if reset_init_vals:
                init_vals = model_.par.x0.copy()            

        if reset_init_vals:
            model_.par.x0 = init_vals.copy()
        model_.par.HH_type = HH_type
        if isinstance(paramvals,dict):
            for key in paramvals:
                setattr(model_.par, key, paramvals[key][i])
            if N_params > 1:
                paramstr = ' = '.join(paramlist)
            else:
                paramstr = paramlist      
            paramlist_dict[i] = [paramvals[x][i] for x in paramvals]   
        else:
            orgvals = [getattr(model_.par,param) for param in paramlist]

            for paramnum,param in enumerate(paramlist):
                if isinstance(paramvals[i],str):
                    setattr(model_.par, param, paramvals[i])
                elif isinstance(getattr(model_.par, param), np.ndarray):
                    setattr(model_.par, param, np.array(paramvals[i]))
                else:
                    try:
                        setattr(model_.par, param, paramvals[i][paramnum])
                    except:
                        setattr(model_.par, param, paramvals[i])
                if N_params > 1:
                    paramstr = ' = '.join(paramlist)
                else:
                    paramstr = paramlist 
            try:
                print(f'{paramstr} = {paramvals[i]:.2f}')
            except:
                try:
                    print(f'{paramstr} = '+paramvals[i])
                except:
                    #print(f'{paramstr} = '+paramvals[i])
                    print([paramlist[x] + '=' + str(paramvals[i][x]) for x in range(N_params)])

        if upars is not None:
            for j in upars:
                setattr(model_.par, j, upars[j])

        # Do stuff
        if homotopic_cont:
            if all(np.isclose(paramvals[i], orgvals[param], rtol = 0.1) for param in range(N_params)):
                model_.find_ss(do_print=False)
            else:
                target_vals = [paramvals[i] for x in paramlist]
                utils.homotopic_cont(model_, paramlist, target_vals, nsteps = 3, noisy=True, orgvals=orgvals)
        else:
            model_.find_ss(do_print=False)
        if model_.par.HH_type=='HA':
            skip_hh = False
            if reuse_HH_jac and compiled:
                skip_hh = True          
        else:
            skip_hh = True
                
        model_.compute_jacs(do_print=False,skip_shocks=True,skip_hh=skip_hh)
        if foreignshock is not True:
            utils.Create_shock(shock, model_, jump,absval=absval)
        else:
            GetForeignEcon.create_foreign_shock(model_, model_foreign)

        #model_.find_transition_path(do_print=False)
        model_.transition_path(do_print=False)
        if scalesize is not None:
            utils.scaleshock(scalevar, model_, size=scalesize)
            scale = getattr(model_.par,'scale')
        else:
            scale = 1.0 
        #scale=1.0 
        print('\n')

        # Get IRFs
        for j in range(len(varlist)):
            if varlist[j] == 'Walras':
                dXs[i,j,:] = utils.get_dX(varlist[j], model_, scaleval=scale, absvalue=True) 
            elif type(varlist[j]) is dict:
                key = list(varlist[j].keys())[0]
                if varlist[j][key][2]: # Cumulative sum
                    num   = utils.get_dX(varlist[j][key][0], model_, scaleval=scale, absvalue=False)
                    denom = utils.get_dX(varlist[j][key][1], model_, scaleval=scale, absvalue=False)                    
                    for t in range (model_.par.T):
                        dXs[i,j,t] = np.sum(num[:8]) / np.sum(denom[:8])
                else:
                    dRel = getattr(model_.path, varlist[j][key][0])[0,:]/getattr(model_.path, varlist[j][key][1])[0,:]
                    Rel_ss = getattr(model_.ss, varlist[j][key][0])/getattr(model_.ss, varlist[j][key][1]) 
                    dXs[i,j,:] = (dRel/Rel_ss-1)*100*model_.par.scale 
            elif np.isclose(getattr(model_.ss, varlist[j]), 0):
                dXs[i,j,:] = utils.get_dX(varlist[j], model_, scaleval=scale, absvalue=True) 
            elif varlist[j] in pctp:
                dXs[i,j,:] = utils.get_dX(varlist[j], model_, scaleval=scale, absvalue=True) * 100
            else:
                dXs[i,j,:] = utils.get_dX(varlist[j], model_, scaleval=scale, absvalue=False) * 100
        compiled=True

    # fig,ax = plt.subplots(1,N_vars,figsize=(4*N_vars,3))
    num = len(varlist) 
    nrows = num//4+1
    ncols = np.fmin(num,4)
    if num%4 == 0: nrows -= 1 
    fig = plt.figure(figsize=(4*ncols,3*nrows))    
    if boolpars:
        alphalist = np.flip(np.linspace(0.4,1,N_models))
        col = ['Darkgreen', 'firebrick']
        for j in range(N_vars):       
            ax = fig.add_subplot(nrows,ncols,j+1)
            for i in range(N_models):
                try:
                    ax.plot(dXs[i,j,:T_max]*100,label=f'{paramstr} = {paramvals[i]}',
                               linewidth=2.3, color=col[i])
                except:  
                    ax.plot(dXs[i,j,:T_max]*100,label=f'{paramstr} = {paramlist_dict[i]}',
                               linewidth=2.3, color=col[i])
                ax.plot(np.zeros(T_max),'--', color='black')             
            ax.set_ylabel('% diff. to s.s.')
            ax.set_title(varlist[j])
        plt.gca().legend()
        fig.tight_layout(pad=1.6)      
    else:
        if alphacolor:
            alphalist = np.flip(np.linspace(0.4,1,N_models))
            col = 'Darkgreen'
        else:
            alphalist = np.flip(np.linspace(1,1,N_models))
            set_palette("colorblind")
            
        for j in range(N_vars):       
            ax = fig.add_subplot(nrows,ncols,j+1)
            noncum = True
            if type(varlist[j]) is dict:
                if varlist[j][list(varlist[j].keys())[0]][2]: noncum = False # Cumulative

            if noncum:
                ax.plot(np.zeros(T_max),'--', color='black')  
                ax.set_xlabel('Quarters', fontsize=14)
                kk=0
                for i in range(N_models):
                    if not alphacolor:
                        if i ==3:
                            kk=0
                        col = 'C' + str(kk+i)
                    if modellabels is not None:
                        label = modellabels[i]
                    else:
                        try:
                            label = f'{paramstr} = {paramvals[i]}'
                        except:
                            label = f'{paramstr} = {paramlist_dict[i]}'
                    
                    ax.plot(dXs[i,j,:T_max],label=label,
                               alpha=alphalist[i], linewidth=2.3, color=col)
                if varlist[j] in pctp:
                    ax.set_ylabel('Pct. points diff. to s.s.')
                else:
                    ax.set_ylabel('% diff. to s.s.')
                if varlistlabels is not None:
                    ax.set_title(varlistlabels[j])
                elif type(varlist[j]) is dict:
                    key = list(varlist[j].keys())[0]
                    ax.set_title(key)
                    if varlist[j][key][2]:
                        try:
                            if np.min(dXs[:,j,0]) > 0:
                                ax.set_ylim([0.0, np.max(dXs[:,j,0])*2])
                            else:
                                ax.set_ylim([np.min(dXs[:,j,0])*1.5, np.max(dXs[:,j,0])*1.5])
                        except:
                            pass

                else:
                    ax.set_title(varlist[j])
                if j == legendplacement:
                    ax.legend(frameon=True, prop={'size': 10})
            else:
                ax.axhline(y=0.0,linewidth=1, color='black')
                ax.set_ylabel('Cumulative multiplier')
                key = list(varlist[j].keys())[0]
                ax.set_title(key)                
                for i in range(N_models):
                    if modellabels is not None:
                        label = modellabels[i]
                    else:
                        try:
                            label = f'{paramstr} = {paramvals[i]}'
                        except:
                            label = f'{paramstr} = {paramlist_dict[i]}'
                    bars=ax.bar(label, dXs[i,j,0],  width = 0.4)
                    ax.bar_label(bars,padding=0.3, fmt='%.2f')
                    maxy = np.max(abs(dXs[:,j,0]))*1.5
                    if min(dXs[:,j,0]) > 0:
                        miny = 0
                    else:
                        miny = -1.5* maxy
                ax.set_ylim([miny,maxy])

        if title is not None: fig.suptitle(title, fontsize=18)
        fig.tight_layout(pad=1.6)
        
    return fig, dXs



def plot_paths(modeldict, modellist, varlistplot, T_max, modellabel, labels=None, 
               savetitle=None, markerstyle=None, pctp=[], lstyle=None, customscale={}, title=None):
    num = len(varlistplot)
    maxrow = 3
    nrows = num//maxrow+1
    ncols = np.fmin(num,maxrow)
    numlegend = 2

    fig = plt.figure(figsize=(3.7*ncols,3.0*nrows))
    for i in range(len(varlistplot)):
        var = varlistplot[i]
        ax = fig.add_subplot(nrows,ncols,i+1)
        ax.set_title(var)  
        for numj,j in enumerate(modellist):  
            if lstyle is not None:
                lstyle_ = lstyle[numj]
            else:
                lstyle_= '-'
            ssval = getattr(modeldict['ss'],var)
            path = modeldict[j][var]
            scale =  modeldict[j]['scale']
            if var in customscale:
                dX = scale * (path[:T_max]) * 100  / (customscale[var])
                #ax.plot(np.arange(T_max),dpathvalue, label=modellabel[numj], linewidth=1.9, marker=markerstyle[numj], linestyle=lstyle_)
                if 'descr' in customscale:
                    ax.set_ylabel(customscale['descr'])               
            elif np.isclose(ssval,0): 
                dX = scale * (path[:T_max]) * 100
                ax.set_ylabel('Pct. points diff. to s.s.')
            elif var in pctp:
                dX = scale * (path[:T_max]-ssval) * 100
                ax.set_ylabel('Pct. points diff. to s.s.')
            else:
                dX = scale * (path[:T_max]/ssval-1) * 100
                ax.set_ylabel('% diff. to s.s.')
            ax.plot(np.arange(T_max), dX, label=modellabel[numj], linewidth=1.9, marker=markerstyle[numj], linestyle=lstyle_)
            ax.plot(np.arange(T_max), np.zeros(T_max), '--', linewidth=1.4, color='black')

        ax.set_xlabel('Quarters', fontsize=13)
        if labels is not None:
            ax.set_title(labels[i])
        else:
            ax.set_title(var)
        if i == numlegend:
            ax.legend(frameon=True, prop={'size': 10})
    if title is not None:
        fig.suptitle(title, fontsize=16)

    fig.tight_layout(pad=1.6)
    if savetitle is not None:
        fig.savefig(savetitle,bbox_inches='tight') 
    plt.show()       
    return fig 