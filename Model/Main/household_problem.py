# solving the household problem
import numpy as np
import numba as nb
from utils import isclose
from consav.linear_interp import interp_1d_vec
import grids


@nb.njit(parallel=True, cache=False)
def solve_hh_backwards(par, ra, UniformT, LT, wnT, wnNT, eps_beta, z_trans, vbeg_a_plus, vbeg_a, a, c, c_T, c_NT, inc_T,inc_NT):
    """ solve backwards with Va_p from previous iteration """

    # b. EGM loop
    for i_beta in nb.prange(par.Nbeta):
        for i_z in nb.prange(par.Nz):
            if par.s_set[i_z] == 0:
                w = wnT
                T = UniformT * par.sT_vec[0]
            elif par.s_set[i_z] == 1:
                w = wnNT
                T = UniformT * par.sT_vec[1]

            m = (1 + ra) * par.a_grid + w * par.z_grid_ss[i_z] - LT * par.LT_grid[i_z] + T

            # i. EGM
            c_endo = (eps_beta * (par.beta_grid[i_beta]) * vbeg_a_plus[i_beta, i_z]) ** (-1 / par.CRRA)
            m_endo = c_endo + par.a_grid

            # interpolation
            interp_1d_vec(m_endo, par.a_grid, m, a[i_beta, i_z, :])

            # enforce borrowing constraint
            a[i_beta, i_z, :] = np.fmax(a[i_beta, i_z, :], par.a_min)
            c[i_beta, i_z, :] = m - a[i_beta, i_z, :]

        # b. expectation step
        va = (1 + ra) * c[i_beta] ** (-par.CRRA)
        vbeg_a[i_beta] = z_trans[i_beta] @ va

    # extra output
    c_T[:, :par.Ne, :] = c[:, :par.Ne, :]
    c_NT[:, par.Ne:, :] = c[:, par.Ne:, :]

    for i_z in nb.prange(par.Nz):
        if par.s_set[i_z] == 0:
            inc_T[:, i_z, :] = a[:, i_z, :] * ra + wnT - LT * par.LT_grid[i_z] + UniformT * par.sT_vec[0]
        elif par.s_set[i_z] == 1:
            inc_NT[:, i_z, :] = a[:, i_z, :] * ra + wnNT - LT * par.LT_grid[i_z] + UniformT * par.sT_vec[1]


@nb.njit
def util(par, c, N, psi):
    if isclose(par.sigma, 1.0):
        u = np.log(c) - psi * (N) ** (1 + par.inv_frisch) / (1 + par.inv_frisch)
    else:
        u = (c) ** (1 - par.CRRA) / (1 - par.CRRA) - psi * (N) ** (1 + par.inv_frisch) / (1 + par.inv_frisch)

    return u


def compute_RA_jacs(ss, par):
    U = np.triu(np.ones((par.T, par.T)), k=0)
    beta = par.beta_mean
    for j in range(par.T):
        par.M_Y[j, :] = (1 - beta) * beta ** np.arange(par.T)

    par.M_R[:] = -1 / par.CRRA * (np.eye(par.T) - par.M_Y) @ U * ss.CR * beta
    par.M_beta[:] = -1 / par.CRRA * (np.eye(par.T) - par.M_Y) @ U * (1 + ss.r) * ss.CR

    if par.HH_type == 'TA-IM':
        par.M_R[:] *= (1 - par.Agg_MPC)
        par.M_Y[:] *= (1 - par.Agg_MPC)
        par.M_Y[:] += par.Agg_MPC * np.eye(par.T)


def prepare_hh_ss(model):
    """ prepare the household block for finding the steady state """

    par = model.par
    ss = model.ss

    ############
    # 1. grids #
    ############

    grids.create_grids_full(model)

    #############################################
    # 2. transition matrix initial distribution #
    #############################################

    for i_beta in range(par.Nbeta):
        ss.Dz[i_beta, :] = par.z_ergodic_ss / par.Nbeta
        ss.Dbeg[i_beta, :, 0] = ss.Dz[i_beta, :]
        ss.Dbeg[i_beta, :, 1:] = 0.0

        ################################################
    # 3. initial guess for intertemporal variables #
    ################################################

    # Initial values for HH problem
    useinitvals = False
    if useinitvals == True:
        try:
            par.init_vals_HH = np.load('saved/Va_init.npz')['Va_init']
        except:
            par.init_vals_HH = np.load('main/saved/Va_init.npz')['Va_init']

        if par.init_vals_HH.shape != model.ss.c.shape:
            par.init_vals_HH = None

    Va = np.zeros((par.Nfix, par.Nz, par.Na))
    m = (1 + ss.ra) * par.a_grid[np.newaxis, :] + par.z_grid_ss[:, np.newaxis] + ss.UniformT
    if useinitvals:
        model.Va[:, :, :] = par.init_vals_HH.copy()
    else:
        a = 0.90 * m  # pure guess
        c = m - a
        Va[:, :, :] = (1 + ss.ra) * c[np.newaxis, :, :] ** (-par.CRRA)

    for i_fix in range(par.Nfix):
        ss.vbeg_a[i_fix] = ss.z_trans[i_fix] @ Va[i_fix]
