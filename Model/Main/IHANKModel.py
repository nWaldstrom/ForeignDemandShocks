import numpy as np
from EconModel import EconModelClass
from GEModelTools import GEModelClass
from steady_state import calibrate_ss
from transition_path_blocks import block_pre, block_post
from household_problem import solve_hh_backwards, prepare_hh_ss


class HANKModelClass(EconModelClass, GEModelClass):

    #########
    # setup #
    #########      

    def settings(self):
        """ fundamental settings """

        # a. namespaces
        self.namespaces = ['par', 'sol', 'sim', 'ss', 'path', 'ini']

        # b. other attributes (to save them)
        self.other_attrs = ['grids_hh', 'pols_hh', 'inputs_hh', 'inputs_exo', 'unknowns', 'targets', 'varlist_hh',
                            'varlist', 'jac']

        # household
        self.grids_hh = ['a']  # grids
        self.pols_hh = ['a']  # policy functions
        self.inputs_hh_z = []  # transition matrix inputs

        self.inputs_hh = ['ra', 'wnT', 'wnNT', 'UniformT', 'LT', 'eps_beta']  # inputs to household problem
        self.intertemps_hh = ['vbeg_a']  # inputs to household problem
        self.outputs_hh = ['a', 'c', 'c_T', 'c_NT', 'inc_T', 'inc_NT']  # output of household problem

        # GE
        self.shocks = ['C_s', 'piF_s', 'iF_s', 'G_exo', 'eps_beta', 'UniformT_exo', 'di', 'FD', 'ND', 'VAT',
                       'subP']  # exogenous inputs
        self.unknowns = ['pi', 'ZT', 'ZNT', 'NFA', 'B', 'piF', 'piH_s']  # endogenous inputs

        self.targets = ['goods_mkt_T', 'goods_mkt_NT', 'NKWPCT',
                        'NKWPCNT', 'NFA_target',
                        'Taylor', 'G_budget']  # targets

        self.varlist = [  # all variables
            # Quantities 
            'ZNT', 'ZT', 'Z',
            'U_hh', 'U_NT_hh', 'U_T_hh',
            'A_hh', 'C_hh', 'C_T_hh', 'C_NT_hh', 'ND', 'INC_T_hh', 'INC_NT_hh',
            'Y', 'YT', 'YNT',
            'GDP_T', 'GDP_NT', 'GDP',
            'XT2NT', 'XNT2T', 'XNT2NT', 'XT2T', 'XT', 'XNT', 'XM2T',
            'PXT', 'PXNT',
            'PO_T', 'PO_NT', 'OT', 'ONT', 'PGDP', 'FD',
            'eps_beta',
            'C_s',
            'CHtM',
            'CR',
            'G_exo', 'G_trans', 'subP', 'VAT',
            'A',
            'N', 'NT', 'NNT',
            'C',
            'CF',
            'CH',
            'CH_s',
            'CT',
            'CNT',
            'B', 'G', 'LT', 'UniformT', 'G_T', 'G_NT', 'UniformT_exo',
            'C_T', 'C_NT',
            'Div', 'DivT', 'DivNT',
            'NX', 'Imports', 'Exports',
            'NFA',
            # prices
            'P', 'PT', 'PNT', 'PTP',
            'PH',
            'PF',
            'PH_s',
            'PF_s',
            'DomP',
            'rF_s',
            'wnT', 'wnNT',
            'E',
            'pD',
            'WT', 'WNT', 'w', 'Income',
            'Q',
            'wT',
            'wNT',
            'mcT',
            'mcNT',
            # Inflation
            'piNT',
            'piF',
            'piF_s',
            'piH_s',
            'pi',
            'ppi',
            'piH',
            'piWT',
            'piWNT',
            'piW',
            # interest rates and returns 
            'di',
            'r',
            'i',
            'iF_s',
            'ra',
            'UC', 'UCT', 'UCNT',
            'UC_T', 'UC_NT',
            # misc 
            'Walras',
            # targets 
            'goods_mkt',
            'NKWPCT', 'NKWPCNT',
            'Taylor',
            'goods_mkt_NT', 'goods_mkt_T',
            'NFA_target',
            'piF_target',
            'piH_s_target',
            'exports_clearing', 'PY_T',
            'G_budget',
            'ToT',
        ]

        # functions
        self.solve_hh_backwards = solve_hh_backwards
        self.block_pre = block_pre
        self.block_post = block_post

    def setup(self):
        """ set baseline parameters """

        par = self.par

        # Preferences
        par.inv_frisch = 2.0  # Frisch elasticity = 1/2 = 0.5
        par.CRRA = 2.0  # CRRA coefficient. Corresponds to EIS = 0.5
        par.beta_mean = np.nan  # discount factor, mean, range is [mean-width,mean+width]
        par.beta_delta = np.nan  # discount factor, width, range is [mean-width,mean+width]

        par.Nbeta = 3  # discount factor, number of states
        par.Nfix = par.Nbeta

        # Earnings states and MPCs
        par.rho_e = 0.95  # AR(1) parameter
        par.sigma_e = 0.25  # std. of persistent shock
        par.Ne = 6  # number of productivity states
        par.Ns = 2  # number of sectors households can be employed in
        par.Nz = par.Ne * par.Ns  # Total number of earnings states
        par.sT = 0.30  # share of households in tradable sector 
        par.Agg_MPC = 0.55  # Aggregate MPC - target in calibration
        par.psi = np.array([np.nan, np.nan])  # Disutility of work - calibrated to solve NKPWC in steady state
        par.iF_s_exo = 0.005  # target for steady state real interest rate

        # Asset grids          
        par.a_min = 0.0  # lower limit of assets - 0 is no borrowing
        par.a_max = 2000  # maximum point in grid for a
        par.Na = 300  # number of grid points
        par.W2INC_target = 10.  # Steady state target for household wealth to GDP

        # Production and markups 
        par.TFP_s = np.array([np.nan, np.nan])  # sector specific productivity
        par.TFP = np.nan  # aggregate productivity
        par.sigma_NX = 1.0  # elasticity of substitution between labor and intermediates. 1 is Cobb-Douglas
        par.alphaX = np.array([np.nan, np.nan])  # output elasticity of intermediate goods aggregate in production

        par.X_expshare = np.array([0.8, 0.55])

        par.HH_importshare = 0.4  # Firm imports make up 60% of total imports
        par.X_share_T = np.array([np.nan, np.nan, np.nan])  # First element is T2T, second is NT2T, third is M2T
        par.X_share_NT = np.array([0.60, 1 - 0.60])  # First element is NT2NT, second is T2NT
        par.etaX = 0.5  # elasticity of substitution between different intermediate goods
        par.prodbeta = np.array([np.nan, np.nan])  # parameters in production function
        par.prodalpha = np.array([np.nan, np.nan])  # parameters in production function

        # Markups and NPCs
        par.mu = np.array([np.nan, np.nan])  # markups
        par.mu[:] = 1.1
        par.FixedCost = np.array([np.nan, np.nan])

        par.epsilon_w = 11.0  # Elasticity of substitution determining wage markup
        par.epsilon = par.mu / (par.mu - 1)  # Elasticity of substitution in demand for goods

        par.NKslope = np.array([np.nan, np.nan])
        par.NKslope[:] = 0.15  # Slope of NKPC
        par.NKWslope = np.array([np.nan, np.nan])
        par.NKWslope[:] = 0.03  # Slope of NKWPC

        par.theta_w = par.epsilon / par.NKWslope  # Rotermberg parameter for wages  - calibrated
        par.theta = np.array([np.nan, np.nan])  # Rotermberg parameter for prices  - calibrated

        # Monetary policy
        par.MonPol = 'Taylor'  # Monetary policy rule
        par.TaylorType = 'ppi'  # Target in Taylor rule
        par.floating = True  # Floating or fixed exchange rate
        par.phi = 1.5  # Taylor rule coefficient
        par.phi_back = 0.85  # Degree of interest rate smoothing in Taylor rule

        # Public sector
        par.G_GDP_ratio = 0.17  # G/GDP
        par.B_GDP_ratio = 0.95 * 4  # B/GDP (annual)
        par.sGT_ss = 0.2  # Share of public consumption going to tradeables in steady state
        par.sGT = par.sGT_ss  # Share of public consumption going to tradeables when shocking G_eps
        par.sTransferT = 0.5  # Share of transfers going to tradeables (Note: not exactly because less households are employed in tradeables)
        par.sT_vec = np.array([1 + par.sTransferT, 1 - par.sTransferT])
        par.tauB = 50  # Finance expenditures using taxes after tauB number of quarters
        par.deltaB = 20
        par.epsB = 0.5
        par.debt_rule = False  # Use public debt rule from Auclert et al 2021
        par.FD_shock = True  # If true then VAT and subP are not exogenous but depend on FD
        par.VAT_weight = 0.446975  # Weight that minimizes difference between ND and FD shocks

        # Shock specifications 
        par.beta_corr = 0.0
        par.rho = 0.8  # common persistence
        for var in self.shocks:
            setattr(par, 'jump_' + var, 0.01)
            setattr(par, 'rho_' + var, par.rho)

        # Trade parameters 
        par.X2Y_target = 0.36
        par.alpha = np.nan
        par.alphaT = 0.41
        par.alpha_F = np.nan
        par.gamma = 1.5
        par.eta = 1.5
        par.etaT = 1.5

        # debt elastic interest rate for RA models
        par.r_debt_elasticity = 0.0001

        # Misc.
        par.T = 300  # length of path
        par.simT = 50000
        par.max_iter_solve = 50_000  # maximum number of iterations when solving
        par.max_iter_simulate = 50_000  # maximum number of iterations when simulating
        par.max_iter_broyden = 100  # maximum number of iteration when solving eq. system

        par.tol_solve = 1e-11  # tolerance when solving
        par.tol_simulate = 1e-12  # tolerance when simulating
        par.tol_broyden = 1e-9  # tolerance when solving eq. system
        par.scale = np.nan  # Used to scale shocks for plots
        par.ModelForeignEcon = True

        # initial values for steady state solvers 
        par.x0 = np.array([0.13948243, 0.14228652, 1.64951897, 1.67570136,
                           0.20677909])  # Initial values for steady state calibration
        par.x0_het = np.array([0.9715154, 0.02300001])  # Initial values for steady state calibration

        # HH type
        par.HH_type = 'HA'
        par.use_RA_jac = True
        par.HA_PooledInc = False
        par.No_HA_list = np.array(['RA-CM', 'RA-IM', 'TA-CM', 'TA-IM'])  # RANK/TANK model labels

        # Containers for RANK Jacobians
        par.M_Y = np.zeros((par.T, par.T))
        par.M_R = np.zeros((par.T, par.T))
        par.M_beta = np.zeros((par.T, par.T))

    def allocate(self):
        """ allocate model """

        par = self.par
        self.create_s_grids()
        self.allocate_GE()

    prepare_hh_ss = prepare_hh_ss

    def HA_PooledInc(self):
        """ One sector """
        self.par.HA_PooledInc = True

    def create_s_grids(self):
        par = self.par
        # create e grids
        par.e_grid_ss = np.zeros(par.Ne)
        par.e_trans_ss = np.zeros([par.Ne, par.Ne])
        par.e_ergodic_ss = np.zeros(par.Ne)
        par.z_ergodic_ss = np.zeros([par.Ns * par.Ne])
        par.z_grid_ss = np.zeros([par.Ns * par.Ne])
        par.s_trans_ss = np.zeros([par.Ns, par.Ns])
        par.s_ergodic_ss = np.zeros(par.Ns)
        par.beta_grid = np.zeros(par.Nbeta)
        par.LT_grid = np.zeros(par.Ns * par.Ne)
        par.LT_grid_path = np.zeros([par.T, par.Ns * par.Ne])
        par.propIfac = np.zeros(par.T)
        par.s_set = np.zeros(par.Ns * par.Ne)
        par.T_NT_trans = np.zeros((2, 2))

    def use_FD_shock(self, FD_shock=True):
        if FD_shock:
            self.par.FD_shock = True
            shocks = [x for x in self.shocks if x not in ['VAT', 'subP', 'FD']]
            shocks += ['FD']
        else:
            self.par.FD_shock = False
            shocks = [x for x in self.shocks if x not in ['VAT', 'subP', 'FD']]
            shocks += ['VAT', 'subP']
        self.update_aggregate_settings(unknowns=None, targets=None, shocks=shocks)

    def HH_type(self, HH_type='HA'):
        """ Use HA or RA/TA model """

        if HH_type == 'HA':
            self.unknowns = [x for x in self.unknowns if x not in ['NFA']]
            self.targets = [x for x in self.targets if x not in ['NFA_target']]
        elif HH_type in self.par.No_HA_list:
            self.unknowns = [x for x in self.unknowns if x not in ['NFA']]
            self.unknowns += ['NFA']
            self.targets = [x for x in self.targets if x not in ['NFA_target']]
            self.targets += ['NFA_target']
            self.inputs_hh = []
            self.outputs_hh = []
            self.pols_hh = []
        else:
            raise ValueError('Incorrect HH type chosen!')

        # Update settings 
        self.update_aggregate_settings(unknowns=self.unknowns, targets=self.targets)

    def find_ss(self, do_print=False):
        calibrate_ss(self, do_print)

    def transition_path(self, do_print, do_end_check=False):
        shock_specs = {}
        for shock in self.shocks:
            shock_specs['d' + shock] = getattr(self.path, shock)[0, :] - getattr(self.ss, shock)
        self.find_transition_path(shock_specs=shock_specs, do_print=do_print, do_end_check=do_end_check)
