import numpy as np
import numba as nb
from EconModel import EconModelClass
from GEModelTools import GEModelClass, lag, lead
import helpers


class NKModelClass(EconModelClass,GEModelClass):
    
    #########
    # setup #
    #########      

    def settings(self):
        """ fundamental settings """

        # a. namespaces
        self.namespaces = ['par','sol','sim','ss','path', 'ini']
        
        # b. other attributes (to save them)
        self.other_attrs = ['grids_hh','pols_hh','inputs_hh','inputs_exo','inputs_endo','targets','varlist_hh','varlist','jac']

        # household
        self.grids_hh = [] # grids
        self.pols_hh = [] # policy functions
        self.inputs_hh_z = [] # transition matrix inputs

        self.inputs_hh = [] # inputs to household problem
        self.outputs_hh = [] # output of household problem
        self.intertemps_hh = []

        self.varlist_hh = [] # variables in household problem

        # GE
        self.shocks = ['iF_s_exo', 'betaF', 'ZF'] # exogenous inputs 
        
        self.unknowns = ['piF_s', 'C_s', 'iF_s'] # endogenous inputs

        self.targets = ['NKPC_res', 'Euler', 'Taylor_res'] # targets
        
        self.varlist = [ # all variables
            'C_s',
            'piF_s',
            'PF_s',
            'rF_s',
            'YF',
            'wF',
            'mcF',
            'NF', 
            'iF_s_exo',
            'iF_s',
            'betaF',
            'ZF',
            'NKPC_res',
            'Euler',
            'Taylor_res'
        ]

    

    def setup(self):
        """ set baseline parameters """

        par = self.par

        # Preferences
        par.CRRA = 2.0 # CRRA coefficient. Corresponds to EIS = 0.5
        par.inv_frisch = 2.0 # Frisch elasticity = 0.5 
        par.vphi = np.nan 

        # Interest rate target 
        par.iF_s_ss = 0.005 # target for real interest rate

        # Business cycle parameters 
        par.kappa = 0.03
        par.muF = 1.1 
        par.phi = 1.5        
        par.MonPol_rule = 'Inf'
        par.phiback = 0.0
        par.pi_index = 0.0
        par.habit = 0.0
        
        # Shock specifications 
        rho = 0.8
        for var in self.shocks:
            setattr(par,'jump_'+var, 0.01)
            setattr(par,'std_'+var, 0.001)
            setattr(par,'rho_'+var, rho)

          
        # Misc.
        par.T = 300 # length of path - should be consistent with T in SOE model           
        par.simT = 50000     
        par.max_iter_solve = 50_000 # maximum number of iterations when solving
        par.max_iter_simulate = 50_000 # maximum number of iterations when simulating
        par.max_iter_broyden = 100 # maximum number of iteration when solving eq. system
        
        par.tol_solve = 1e-10 # tolerance when solving
        par.tol_simulate = 1e-12 # tolerance when simulating
        par.tol_broyden = 1e-9 # tolerance when solving eq. system

        par.Nz = par.Nfix = 1        

        self.solve_hh_backwards = None
        self.block_pre  = For_block_pre
        self.block_post = For_block_post

        
    def allocate(self):
        """ allocate model """

        par = self.par

        # b. solution
        self.allocate_GE()

    def find_ss(model,do_print=False):
        """ find the steady state """

        par = model.par
        sol = model.sol
        sim = model.sim
        ss = model.ss
        
        ss.rF_s = ss.iF_s = par.iF_s_ss
        ss.iF_s_exo = 0.0 
        ss.betaF = 1/(1+ss.rF_s)
        ss.mcF = 1/par.muF
        ss.PF_s = 1.0 
        ss.piF_s = 0.0 
        ss.YF = ss.C_s = ss.NF = 1.0 
        ss.ZF = ss.YF/ss.NF 
        ss.wF = ss.mcF * ss.ZF          
        par.vphi = ss.wF / ss.NF**(par.inv_frisch) / (ss.C_s**(-par.CRRA))  


    def transition_path(self, do_print):
        shock_specs = {}
        for shock in self.shocks:
            shock_specs['d'+shock] = getattr(self.path, shock)[0,:] - getattr(self.ss, shock)
        self.find_transition_path(shock_specs=shock_specs, do_print=do_print, do_end_check=False)                
            

################
# other blocks #
################

@nb.njit(cache=False)
def For_block_pre(par,ini,ss,path,ncols=1):

    for thread in nb.prange(ncols):
        
        C_s = path.C_s[thread,:]
        piF_s = path.piF_s[thread,:]
        PF_s = path.PF_s[thread,:]
        rF_s = path.rF_s[thread,:]
        YF = path.YF[thread,:]
        wF = path.wF[thread,:]
        mcF = path.mcF[thread,:]
        NF = path.NF[thread,:]
        iF_s_exo = path.iF_s_exo[thread,:]
        iF_s = path.iF_s[thread,:]
        betaF = path.betaF[thread,:]
        ZF = path.ZF[thread,:]
        NKPC_res = path.NKPC_res[thread,:]
        Euler = path.Euler[thread,:]
        Taylor_res = path.Taylor_res[thread,:]

        PC_beta = 1/(1+ss.rF_s)

        helpers.P_from_inf(PF_s, piF_s, par.T, ss.PF_s)       

        # Taylor rule
        PF_s_p = lead(PF_s, PF_s[-1])
        piF_s_p = lead(piF_s, ss.piF_s)
        iF_s_lag = lag(ss.iF_s, iF_s)

        if par.MonPol_rule == 'P':
            taylor_rhs = (1 - par.phiback) * (ss.iF_s + par.phi * (np.log(PF_s_p) - np.log(ss.PF_s))) + par.phiback * \
                      iF_s_lag + iF_s_exo
            Taylor_res[:] = iF_s - taylor_rhs
        else:
            taylor_rhs = (1 - par.phiback) * (ss.iF_s + par.phi * piF_s_p) + par.phiback * iF_s_lag + iF_s_exo
            Taylor_res[:] = iF_s - taylor_rhs

        # Fisher
        rF_s[:] = (1 + iF_s) / (1 + piF_s_p) - 1

        # Firms
        YF[:] = C_s
        NF[:] = YF / ZF
        wF[:] = par.vphi * NF ** (par.inv_frisch) / (C_s ** (-par.CRRA))
        mcF[:] = wF / ZF

        # Euler
        C_s_p = lead(C_s, ss.C_s)
        C_s_lag = lag(ss.C_s, C_s)
        Euler[:] = (C_s - par.habit * C_s_lag) ** (-par.CRRA) - (
                    (1 + rF_s) * betaF * (C_s_p - par.habit * C_s) ** (-par.CRRA))

        # NKPC
        piF_s_lag = lag(ss.piF_s, piF_s)
        NKPC_res[:] = piF_s - (
                    par.pi_index / (1 + par.pi_index * PC_beta) * piF_s_lag + par.kappa / (1 + par.pi_index * PC_beta) * (
                        mcF - 1 / par.muF) + piF_s_p * PC_beta / (1 + par.pi_index * PC_beta))

@nb.njit(cache=False)
def For_block_pre_old(par, ini, ss, path, ncols=1):
    for thread in nb.prange(ncols):

        C_s = path.C_s[thread, :]
        piF_s = path.piF_s[thread, :]
        PF_s = path.PF_s[thread, :]
        rF_s = path.rF_s[thread, :]
        YF = path.YF[thread, :]
        wF = path.wF[thread, :]
        mcF = path.mcF[thread, :]
        NF = path.NF[thread, :]
        iF_s_exo = path.iF_s_exo[thread, :]
        iF_s = path.iF_s[thread, :]
        betaF = path.betaF[thread, :]
        ZF = path.ZF[thread, :]
        NKPC_res = path.NKPC_res[thread, :]
        Euler = path.Euler[thread, :]

        PC_beta = 1 / (1 + ss.rF_s)

        helpers.P_from_inf(PF_s, piF_s, par.T, ss.PF_s)

        for t in range(par.T):
            if t == par.T - 1:
                iF_s[t] = ss.iF_s
            else:
                if par.MonPol_rule == 'P':
                    iF_s[t] = (1 - par.phiback) * (
                                ss.iF_s + par.phi * (np.log(PF_s[t + 1]) - np.log(ss.PF_s))) + par.phiback * iF_s[
                                  t - 1] + iF_s_exo[t]
                else:
                    iF_s[t] = (1 - par.phiback) * (ss.iF_s + par.phi * piF_s[t + 1]) + par.phiback * iF_s[t - 1] + \
                              iF_s_exo[t]

        for k in range(par.T):
            t = (par.T - 1) - k
            if t == par.T - 1:  # steady state
                Euler[t] = C_s[t] - ss.C_s
                YF[t] = ss.YF
                rF_s[t] = iF_s[t]
                NKPC_res[t] = piF_s[t] - ss.piF_s
                mcF[t] = ss.mcF
                NF[t] = ss.NF
                wF[t] = ss.wF
            else:
                rF_s[t] = (1 + iF_s[t]) / (1 + piF_s[t + 1]) - 1
                if t > 0:
                    C_lag = C_s[t - 1]
                else:
                    C_lag = ss.C_s
                Euler[t] = (C_s[t] - par.habit * C_lag) ** (-par.CRRA) - (
                            (1 + rF_s[t]) * betaF[t] * (C_s[t + 1] - par.habit * C_s[t]) ** (-par.CRRA))

                YF[t] = C_s[t]
                NF[t] = YF[t] / ZF[t]
                wF[t] = par.vphi * NF[t] ** (par.inv_frisch) / (C_s[t] ** (-par.CRRA))
                mcF[t] = wF[t] / ZF[t]

                if t > 0:
                    pi_lag = piF_s[t - 1]
                elif t == 0:
                    pi_lag = ss.piF_s
                NKPC_res[t] = piF_s[t] - (par.pi_index / (1 + par.pi_index * PC_beta) * pi_lag + par.kappa / (
                            1 + par.pi_index * PC_beta) * (mcF[t] - 1 / par.muF) + piF_s[t + 1] * PC_beta / (
                                                      1 + par.pi_index * PC_beta))


#@nb.njit(cache=False)
def For_block_post(par,ini,ss,path,ncols=1):
    """ evaluate transition path - after household block """

    