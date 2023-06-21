import numpy as np
import numba as nb
from GEModelTools import lag, lead
import helpers
from utils import isclose


###############
#    blocks   #
###############

@nb.njit
def production_block(par, ss, path, thread):
    ZT = path.ZT[thread, :]
    ZNT = path.ZNT[thread, :]
    Z = path.Z[thread, :]
    mcT = path.mcT[thread, :]
    mcNT = path.mcNT[thread, :]

    # intermediate goods 
    XT = path.XT[thread, :]
    XNT = path.XNT[thread, :]
    XNT2T = path.XNT2T[thread, :]
    XT2NT = path.XT2NT[thread, :]
    XNT2NT = path.XNT2NT[thread, :]
    XT2T = path.XT2T[thread, :]
    XM2T = path.XM2T[thread, :]
    PXT = path.PXT[thread, :]
    PXNT = path.PXNT[thread, :]

    subP = path.subP[thread, :]
    VAT = path.VAT[thread, :]
    piH = path.piH[thread, :]
    piNT = path.piNT[thread, :]
    PH = path.PH[thread, :]
    PH = path.PH[thread, :]
    PNT = path.PNT[thread, :]
    PF = path.PF[thread, :]

    WT = path.WT[thread, :]
    WNT = path.WNT[thread, :]
    NT = path.NT[thread, :]
    NNT = path.NNT[thread, :]
    N = path.N[thread, :]

    PGDP = path.PGDP[thread, :]
    GDP = path.GDP[thread, :]
    YT = path.YT[thread, :]
    YNT = path.YNT[thread, :]
    Y = path.Y[thread, :]
    wnT = path.wnT[thread, :]
    wnNT = path.wnNT[thread, :]
    w = path.w[thread, :]
    P = path.P[thread, :]
    wT = path.wT[thread, :]
    wNT = path.wNT[thread, :]

    PC_beta = 1 / (1 + ss.r)

    # Solve NKPCs     
    ZT_p = lead(ZT, ss.ZT)
    ZNT_p = lead(ZNT, ss.ZNT)
    VAT_p = lead(VAT, VAT[-1])
    piH_p = lead(piH, ss.piH)
    piNT_p = lead(piNT, ss.piNT)

    # invert NKPC to get real marginal costs
    mcT[:] = (par.theta[0] * piH - (
                PC_beta * (par.theta[0] * piH_p * ZT_p / ZT * (1 - VAT_p) / (1 - VAT)) + (1 - par.epsilon[0]))) / \
             par.epsilon[0]
    mcNT[:] = (par.theta[1] * piNT - (
                PC_beta * (par.theta[1] * piNT_p * ZNT_p / ZNT * (1 - VAT_p) / (1 - VAT)) + (1 - par.epsilon[1]))) / \
              par.epsilon[1]

    # CES prices 
    PXT[:] = helpers.Price_index_T(PNT, PH, PF, par.etaX, par.X_share_T)
    PXNT[:] = helpers.Price_index_NT(PH, PNT, par.etaX, par.X_share_NT)

    if isclose(par.sigma_NX, 1.0) and ss.XT > 0:  # Cobb-Douglas
        WT[:] = (mcT * PH * par.TFP * par.TFP_s[0] * (1 - VAT) / (
                    (PXT * (1 - VAT) / (par.prodalpha[0])) ** (par.prodalpha[0]))) ** (1 / (1 - par.prodalpha[0])) * (
                            1 - par.prodalpha[0]) / (1 - subP)
        WNT[:] = (mcNT * PNT * par.TFP * par.TFP_s[1] * (1 - VAT) / (
                    (PXNT * (1 - VAT) / (par.prodalpha[1])) ** (par.prodalpha[1]))) ** (1 / (1 - par.prodalpha[1])) * (
                             1 - par.prodalpha[1]) / (1 - subP)
    elif isclose(ss.XT, 0.0):  # Linear production function with labor
        WT[:] = (1 - VAT) * mcT * PH * par.TFP * par.TFP_s[0] / (1 - subP)
        WNT[:] = (1 - VAT) * mcNT * PNT * par.TFP * par.TFP_s[1] / (1 - subP)
    else:  # CES
        WT[:] = ((((1 - VAT) * mcT * PH * par.TFP * par.TFP_s[0]) ** (1 - par.sigma_NX) - par.prodalpha[
            0] ** par.sigma_NX * PXT ** (1 - par.sigma_NX)) / par.prodbeta[0] ** par.sigma_NX) ** (
                            1 / (1 - par.sigma_NX)) / (1 - subP)
        WNT[:] = ((((1 - VAT) * mcNT * PNT * par.TFP * par.TFP_s[1]) ** (1 - par.sigma_NX) - par.prodalpha[
            1] ** par.sigma_NX * PXNT ** (1 - par.sigma_NX)) / par.prodbeta[1] ** par.sigma_NX) ** (
                             1 / (1 - par.sigma_NX)) / (1 - subP)

        # Intermediate goods
    if par.alphaX[0] > 0.0 and par.alphaX[1] > 0.0:
        XT[:] = par.prodalpha[0] ** (par.sigma_NX) * (PXT / (PH * mcT)) ** (-par.sigma_NX) * ZT * (
                    par.TFP * par.TFP_s[0]) ** (par.sigma_NX - 1)
        XNT[:] = par.prodalpha[1] ** (par.sigma_NX) * (PXNT / (PNT * mcNT)) ** (-par.sigma_NX) * ZNT * (
                    par.TFP * par.TFP_s[1]) ** (par.sigma_NX - 1)
        XT2NT[:], XNT2T[:], XNT2NT[:], XT2T[:], XM2T[:] = helpers.get_intermediates(XT, XNT, PXT, PXNT, PH, PNT, PF,
                                                                                    par.X_share_T, par.X_share_NT,
                                                                                    par.etaX)
    else:
        XNT2T[:] = 0.0
        XT2NT[:] = 0.0
        XT2NT[:], XNT2T[:], XNT2NT[:], XT2T[:] = 0.0, 0.0, 0.0, 0.0
        PXT[:], PXNT[:] = 0.0, 0.0

    # Production function 
    exp = (par.sigma_NX - 1) / par.sigma_NX
    if isclose(par.sigma_NX, 1.0):
        NT[:] = (ZT / (par.TFP * par.TFP_s[0] * XT ** par.alphaX[0])) ** (1 / (1 - par.alphaX[0]))
        NNT[:] = (ZNT / (par.TFP * par.TFP_s[1] * XNT ** par.alphaX[1])) ** (1 / (1 - par.alphaX[1]))
    else:
        NT[:] = (((ZT / (par.TFP * par.TFP_s[0])) ** exp - par.prodalpha[0] * XT ** exp) / par.prodbeta[0]) ** (1 / exp)
        NNT[:] = (((ZNT / (par.TFP * par.TFP_s[1])) ** exp - par.prodalpha[1] * XNT ** exp) / par.prodbeta[1]) ** (
                    1 / exp)

        # Value added part of CES tree
    nOT = PH * ZT - PXT * XT - PH * par.FixedCost[0]
    nONT = PNT * ZNT - PXNT * XNT - PNT * par.FixedCost[1]
    N[:] = NT + NNT

    for t in range(par.T):
        nOT_denom = ss.PH * ZT[t] - ss.PXT * XT[t] - ss.PH * par.FixedCost[0]
        nONT_denom = ss.PNT * ZNT[t] - ss.PXNT * XNT[t] - ss.PNT * par.FixedCost[1]
        PGDP[t] = (nOT[t] + nONT[t]) / (nOT_denom + nONT_denom) * ss.PGDP
        GDP[t] = (nOT[t] + nONT[t]) / PGDP[t]

    YT[:] = nOT / PGDP
    YNT[:] = nONT / PGDP
    Y[:] = YT + YNT
    Z[:] = (PH * ZT + PNT * ZNT) / PGDP

    #####################
    #        Wages      #
    #####################
    # Average real wage income     
    wnT[:] = (WT * NT / P) / par.sT
    wnNT[:] = (WNT * NNT / P) / (1 - par.sT)

    # Aggregate real wage income
    wT[:], wNT[:] = WT / P, WNT / P
    w[:] = wT + wNT


@nb.njit
def asset_returns(par, ss, path, thread):
    DivT = path.DivT[thread, :]
    DivNT = path.DivNT[thread, :]
    Div = path.Div[thread, :]
    YT = path.YT[thread, :]
    YNT = path.YNT[thread, :]
    VAT = path.VAT[thread, :]
    wnT = path.wnT[thread, :]
    wnNT = path.wnNT[thread, :]
    subP = path.subP[thread, :]
    P = path.P[thread, :]
    PGDP = path.PGDP[thread, :]
    r = path.r[thread, :]
    pD = path.pD[thread, :]
    ra = path.ra[thread, :]
    pi = path.pi[thread, :]

    DivT[:] = (1 - VAT) * YT * PGDP / P - par.sT * wnT * (1 - subP)
    DivNT[:] = (1 - VAT) * YNT * PGDP / P - (1 - par.sT) * wnNT * (1 - subP)
    Div[:] = DivT + DivNT
    Div_p = lead(Div, Div[-1])

    # Solve pD = (pD_p+Div)/(1+r) for pD
    helpers.sol_backwards_lin(pD, ss.pD, Div_p / (1 + r), 1 / (1 + r), par.T)

    # Return at time 0 of the MIT shock is calculated from initial asset position
    ra[0] = (pD[0] + Div[0] + (1 + ss.i) / (1 + pi[0]) * ss.B) / ss.A - 1
    # For the remaining periods arbitrage-conditions hold so 
    ra[1:] = r[:-1]


@nb.njit
def public_finances(par, ss, path, thread):
    VAT = path.VAT[thread, :]
    YT = path.YT[thread, :]
    YNT = path.YNT[thread, :]
    P = path.P[thread, :]
    PGDP = path.PGDP[thread, :]
    subP = path.subP[thread, :]
    WT = path.WT[thread, :]
    NT = path.NT[thread, :]
    WNT = path.WNT[thread, :]
    NNT = path.NNT[thread, :]
    UniformT_exo = path.UniformT_exo[thread, :]
    UniformT = path.UniformT[thread, :]
    G_trans = path.G_trans[thread, :]
    G_exo = path.G_exo[thread, :]
    G_T = path.G_T[thread, :]
    G_NT = path.G_NT[thread, :]
    G = path.G[thread, :]
    PH = path.PH[thread, :]
    PNT = path.PNT[thread, :]
    B = path.B[thread, :]
    i = path.i[thread, :]
    pi = path.pi[thread, :]
    LT = path.LT[thread, :]
    G_budget = path.G_budget[thread, :]

    TR = VAT * (YT + YNT) * PGDP / P - subP * (WT * NT + WNT * NNT) / P
    UniformT[:] = UniformT_exo
    G_trans[:] = G_exo
    G_T[:] = par.sGT_ss * ss.G + par.sGT * G_trans
    G_NT[:] = (1 - par.sGT_ss) * ss.G + (1 - par.sGT) * G_trans
    G[:] = (PH * G_T + PNT * G_NT) / P

    B_lag = lag(ss.B, B)
    i_lag = lag(ss.i, i)

    if par.debt_rule:  # Auclert et al debt rule
        rho_b, t_fin = 0.95, 50
        LT[:] = (G + UniformT - TR + (1 + i_lag) / (1 + pi) * B_lag) - B
        G_budget[:t_fin] = B[:t_fin] - ss.B - rho_b * (
                    B_lag[:t_fin] - ss.B + G[:t_fin] - ss.G + UniformT_exo[:t_fin] - ss.UniformT_exo + (
                        (1 + i_lag[:t_fin]) / (1 + pi[:t_fin]) - 1) * B_lag[:t_fin] - ss.B * ss.r)
        G_budget[t_fin:] = B[t_fin:] - ss.B - rho_b * (
                    B_lag[t_fin:] - ss.B + G[t_fin:] - ss.G + UniformT_exo[t_fin:] - ss.UniformT_exo)
    else:  # Complete debt financed policy for the first tauB periods
        if ss.B > 0:
            for t in range(par.T):
                if t < par.tauB:
                    LT[t] = ss.LT
                elif t <= par.tauB + par.deltaB:
                    ttilde = (t - par.tauB) / par.deltaB
                    tautilde = ss.LT * (B[t - 1] / ss.B) ** par.epsB
                    omega_x = 3 * ttilde ** 2 - 2 * ttilde ** 3
                    LT[t] = (1 - omega_x) * ss.LT + omega_x * tautilde
                else:
                    LT[t] = ss.LT * (B[t - 1] / ss.B) ** par.epsB
        else:
            LT[:] = ss.LT

        G_budget[:] = B + LT - (G + UniformT - TR + (1 + i_lag) / (1 + pi) * B_lag)


@nb.njit
def simple_HHs(par, ss, path, thread):
    wnNT = path.wnNT[thread, :]
    wnT = path.wnT[thread, :]
    UniformT = path.UniformT[thread, :]
    LT = path.LT[thread, :]
    Income = path.Income[thread, :]

    C = path.C[thread, :]
    r = path.r[thread, :]
    eps_beta = path.eps_beta[thread, :]
    CHtM = path.CHtM[thread, :]
    CR = path.CR[thread, :]
    iF_s = path.iF_s[thread, :]
    piF = path.piF[thread, :]
    Q = path.Q[thread, :]
    C_s = path.C_s[thread, :]
    ra = path.ra[thread, :]
    A = path.A[thread, :]
    UC_T = path.UC_T[thread, :]
    UC_NT = path.UC_NT[thread, :]

    Income[:] = wnNT * (1 - par.sT) + wnT * par.sT - LT + UniformT

    if par.HH_type != 'HA':
        if par.use_RA_jac:
            C[:] = ss.C
            C[:] += par.M_Y @ (Income - ss.Income)
            C[:] += par.M_R @ (r - ss.r)
            C[:] += par.M_beta @ (eps_beta - ss.eps_beta) * par.beta_mean
        else:
            if par.HH_type == 'RA-CM' or par.HH_type == 'RA-IM':
                sHtM = 0
            else:
                sHtM = par.Agg_MPC
            for tt in range(par.T):
                t = par.T - 1 - tt
                if par.HH_type == 'RA-CM' or par.HH_type == 'TA-CM':
                    if par.HH_type == 'TA-CM':
                        CHtM[t] = Income[t]
                    else:
                        CHtM[t] = 0.0
                    if t == par.T - 1:
                        CR[t] = ss.CR
                    else:
                        rF_s = iF_s[t] - piF[t]
                        CR[t] = CR[t + 1] * (Q[t + 1] / Q[t] * (1 + rF_s) / (1 + ss.r)) ** (-1 / par.CRRA) * C_s[t] / \
                                C_s[t + 1]
                    C[t] = CR[t] * (1 - sHtM) + sHtM * CHtM[t]
                elif par.HH_type == 'RA-IM' or par.HH_type == 'TA-IM':
                    if par.HH_type == 'TA-IM':
                        CHtM[t] = Income[t]
                    else:
                        CHtM[t] = 0.0
                    if t == par.T - 1:
                        CR[t] = ss.CR
                    else:
                        CR[t] = ((1 + ra[t + 1]) * eps_beta[t] * par.beta_mean * CR[t + 1] ** (-par.CRRA)) ** (
                                    -1 / par.CRRA)
                C[t] = CR[t] * (1 - sHtM) + sHtM * CHtM[t]

        UC_T[:] = C[:] ** (-par.CRRA)
        UC_NT[:] = C[:] ** (-par.CRRA)
        helpers.Get_HH_A(A, C, Income, ra, par.T, ss)


@nb.njit
def mon_pol(par, ss, path, thread):
    PNT = path.PNT[thread, :]
    PH = path.PH[thread, :]
    DomP = path.DomP[thread, :]
    ppi = path.ppi[thread, :]
    pi = path.pi[thread, :]
    piNT = path.piNT[thread, :]
    E = path.E[thread, :]
    ND = path.ND[thread, :]
    Z = path.Z[thread, :]
    i = path.i[thread, :]
    di = path.di[thread, :]
    Taylor = path.Taylor[thread, :]

    i_lag = lag(ss.i, i)
    pi_p = lead(pi, ss.pi)

    # Calculate domestic price index using Paasche price index (sum of NT and H)
    DomP[:] = (PNT * ss.CNT + PH * ss.CH) / (ss.PNT * ss.CNT + ss.PH * ss.CH)
    ppi[:], ppi_p = helpers.Inf(DomP, ss.DomP)

    if par.floating:
        if par.MonPol == 'Taylor':
            if par.TaylorType == 'CPI':
                Taylor_pi = pi
            elif par.TaylorType == 'NT':
                Taylor_pi = lead(piNT, ss.piNT)
            elif par.TaylorType == 'ppi':
                Taylor_pi = ppi
            elif par.TaylorType == 'DomP':
                Taylor_pi = np.log(DomP) - np.log(ss.DomP)
            elif par.TaylorType == 'FoF':
                Taylor_pi = ppi_p + 0.3 / par.phi * np.log(E / lag(ss.E, E))
            elif par.TaylorType == 'Y':
                Taylor_pi = ppi_p + 0.25 / par.phi * np.log(Z / ss.Z)
            Taylor[:] = i - ((ss.i + par.phi * Taylor_pi) * (1 - par.phi_back) + par.phi_back * i_lag + di)
        else:
            Taylor[:] = 1 + ss.r - (1 + i + di) / (1 + pi_p)
    else:
        Taylor[:] = E - ND


@nb.njit
def pricing(par, ss, path, thread):
    PF_s = path.PF_s[thread, :]
    P = path.P[thread, :]
    pi = path.pi[thread, :]
    piF_s = path.piF_s[thread, :]
    piF = path.piF[thread, :]
    PF = path.PF[thread, :]
    PH_s = path.PH_s[thread, :]
    piH_s = path.piH_s[thread, :]
    rF_s = path.rF_s[thread, :]
    iF_s = path.iF_s[thread, :]
    E = path.E[thread, :]
    VAT = path.VAT[thread, :]
    Q = path.Q[thread, :]
    r = path.r[thread, :]
    i = path.i[thread, :]
    NFA = path.NFA[thread, :]
    PH = path.PH[thread, :]
    PT = path.PT[thread, :]
    PNT = path.PNT[thread, :]
    piNT = path.piNT[thread, :]
    ToT = path.ToT[thread, :]
    CH_s = path.CH_s[thread, :]
    C_s = path.C_s[thread, :]
    piH = path.piH[thread, :]

    # Prices and inflation 
    piF_s_p = lead(piF_s, ss.piF_s)
    helpers.P_from_inf(PF_s, piF_s, par.T, ss.PF_s)
    helpers.P_from_inf(P, pi, par.T, ss.P)  # P
    helpers.P_from_inf(PF, piF, par.T, ss.PF)  # PF
    helpers.P_from_inf(PH_s, piH_s, par.T, ss.PH_s)  # PH_s
    pi_p = lead(pi, ss.pi)
    rF_s[:] = (1 + iF_s) / (1 + piF_s_p) - 1

    # Exchange rate 
    E[:] = PF / PF_s * (1 - VAT)
    Q[:] = E * PF_s / P
    Q_terminal = Q[-1]
    Q_p = lead(Q, Q_terminal)

    # UIP and Fisher
    if par.HH_type == 'HA':
        r[:] = (1 + rF_s) * (Q_p / Q) - 1
    else:
        RP = np.exp(-par.r_debt_elasticity * (NFA / ss.GDP - ss.NFA / ss.GDP))
        r[:] = (1 + rF_s) * (Q_p / Q) * RP - 1
    i[:] = (1 + r) * (1 + pi_p) - 1

    # LCP/PCP 
    PH[:] = PH_s * E / (1 - VAT)
    piH[:], piH_p = helpers.Inf(PH, ss.PH)

    PT[:] = helpers.Price_index(PH, PF, par.eta, par.alpha)
    if isclose(par.etaT, 1.0):
        PNT[:] = (P / PT ** par.alphaT) ** (1 / ((1 - par.alphaT)))
    else:
        PNT[:] = ((P ** (1 - par.etaT) - par.alphaT * PT ** (1 - par.etaT)) / (1 - par.alphaT)) ** (1 / (1 - par.etaT))
    piNT[:], piNT_p = helpers.Inf(PNT, ss.PNT)

    ToT[:] = PF / PH_s / E * (1 - VAT)

    # Exports 
    CH_s[:] = helpers.Armington(PH_s, PF_s, C_s, par.gamma, par.alpha_F)


@nb.njit(cache=False)
def block_pre(par, ini, ss, path, ncols=1):
    """ evaluate transition path """

    for thread in nb.prange(ncols):

        VAT = path.VAT[thread, :]
        subP = path.subP[thread, :]
        FD = path.FD[thread, :]

        # update parameters
        par.theta_w[:] = par.epsilon / par.NKWslope
        par.theta[:] = par.epsilon / par.NKslope

        # Fiscal devaluation
        if par.FD_shock:
            VAT[:] = par.VAT_weight * FD
            subP[:] = (1 - par.VAT_weight) * FD

        #####################
        #  Prices and UIP   #
        #####################

        pricing(par, ss, path, thread)

        #####################
        #     Production    #
        #####################

        production_block(par, ss, path, thread)

        #################################
        #   Dividends and asset returns #
        #################################

        asset_returns(par, ss, path, thread)

        #########################
        #     Public finances   #
        ######################### 

        public_finances(par, ss, path, thread)

        ###############################
        #   RA/TA Household problem   #
        ###############################

        simple_HHs(par, ss, path, thread)

        #########################
        #    Monetary policy    #
        #########################

        mon_pol(par, ss, path, thread)


@nb.njit(cache=False)
def block_post(par, ini, ss, path, ncols=1):
    """ evaluate transition path """

    for thread in nb.prange(ncols):

        # Prices
        P = path.P[thread, :]
        CH_s = path.CH_s[thread, :]
        PT = path.PT[thread, :]
        PF = path.PF[thread, :]
        PNT = path.PNT[thread, :]
        PH = path.PH[thread, :]

        # Production
        ZT = path.ZT[thread, :]
        ZNT = path.ZNT[thread, :]
        GDP_NT = path.GDP_NT[thread, :]
        GDP_T = path.GDP_T[thread, :]

        # intermediate goods 
        XT = path.XT[thread, :]
        XNT = path.XNT[thread, :]
        XNT2T = path.XNT2T[thread, :]
        XT2NT = path.XT2NT[thread, :]
        XNT2NT = path.XNT2NT[thread, :]
        XT2T = path.XT2T[thread, :]
        XM2T = path.XM2T[thread, :]
        PXT = path.PXT[thread, :]
        PXNT = path.PXNT[thread, :]

        # Wages and employment
        NT = path.NT[thread, :]
        NNT = path.NNT[thread, :]
        WNT = path.WNT[thread, :]
        WT = path.WT[thread, :]
        wNT = path.wNT[thread, :]
        wT = path.wT[thread, :]
        piWT = path.piWT[thread, :]
        piWNT = path.piWNT[thread, :]

        # Asset returns
        pD = path.pD[thread, :]
        ra = path.ra[thread, :]

        # Public finances
        G = path.G[thread, :]
        G_T = path.G_T[thread, :]
        G_NT = path.G_NT[thread, :]
        B = path.B[thread, :]

        # Household variables
        UC_T = path.UC_T[thread, :]
        UC_NT = path.UC_NT[thread, :]
        C = path.C[thread, :]
        C_T = path.C_T[thread, :]
        C_NT = path.C_NT[thread, :]
        A = path.A[thread, :]
        CT = path.CT[thread, :]
        CNT = path.CNT[thread, :]
        CF = path.CF[thread, :]
        CH = path.CH[thread, :]
        C = path.C[thread, :]
        A = path.A[thread, :]
        C_hh = path.C_hh[thread, :]
        A_hh = path.A_hh[thread, :]
        C_T_hh = path.C_T_hh[thread, :]
        C_NT_hh = path.C_NT_hh[thread, :]

        # Trade 
        NFA = path.NFA[thread, :]
        NX = path.NX[thread, :]
        Exports = path.Exports[thread, :]
        Imports = path.Imports[thread, :]
        Walras = path.Walras[thread, :]

        # Targets
        NFA_target = path.NFA_target[thread, :]
        NKWPCNT = path.NKWPCNT[thread, :]
        NKWPCT = path.NKWPCT[thread, :]
        goods_mkt_NT = path.goods_mkt_NT[thread, :]
        goods_mkt_T = path.goods_mkt_T[thread, :]

        if par.HH_type == 'HA':
            C_T[:] = C_T_hh
            C_NT[:] = C_NT_hh
            C[:] = C_hh
            A[:] = A_hh
            if par.HA_PooledInc:
                UC_T = C ** (-par.CRRA)
                UC_NT = C ** (-par.CRRA)
            else:
                UC_T = C_T ** (-par.CRRA)
                UC_NT = C_NT ** (-par.CRRA)
        else:
            C_T[:] = C
            C_NT[:] = C

            # Get consumption at lower levels of CES tree
        CT[:], CNT[:] = helpers.CES_demand_T(C, PT, PNT, P, par.etaT, par.alphaT)
        CF[:], CH[:] = helpers.CES_demand(CT, PH, PF, PT, par.eta, par.alpha)

        ######################
        #       Walras       #
        ######################        
        if par.HH_type == 'HA':
            NFA[:] = A - pD - B
        else:
            NFA_target[:] = NFA - (A - pD - B)
        NFA_lag = lag(ss.NFA, NFA)
        NX[:] = (PH * ZT + PNT * ZNT - PXT * XT - PXNT * XNT) / P - C - G - PH / P * par.FixedCost[0] - PNT / P * \
                par.FixedCost[1]
        Walras[:] = NFA - (NX + (1 + ra) * NFA_lag)
        Exports[:] = CH_s
        Imports[:] = CF + XM2T

        ####################################
        # Goods market clearing and NKWPCs #
        ####################################

        GDP_T[:] = ((ZT - par.FixedCost[0]) - XT2NT - XT2T)
        GDP_NT[:] = ((ZNT - par.FixedCost[1]) - XNT2T - XNT2NT)
        goods_mkt_T[:] = GDP_T * PH - (PH * CH + PH * CH_s + PH * G_T)
        goods_mkt_NT[:] = GDP_NT * PNT - (PNT * CNT + PNT * G_NT)

        PC_beta = 1 / (1 + ss.r)
        piWT[:], piWT_p = helpers.Inf(WT, ss.WT)
        piWNT[:], piWNT_p = helpers.Inf(WNT, ss.WNT)

        v_prime = par.psi[0] * (NT / par.sT) ** par.inv_frisch
        LHS_WPC = np.log(1 + piWT)
        RHS_WPC = par.NKWslope[0] * NT * (v_prime - wT * (par.epsilon_w - 1) / par.epsilon_w * UC_T) + PC_beta * np.log(
            1 + piWT_p)
        NKWPCT[:] = LHS_WPC - RHS_WPC

        v_prime = par.psi[1] * (NNT / (1 - par.sT)) ** par.inv_frisch
        LHS_WPC = np.log(1 + piWNT)
        RHS_WPC = par.NKWslope[1] * NNT * (
                    v_prime - wNT * (par.epsilon_w - 1) / par.epsilon_w * UC_NT) + PC_beta * np.log(1 + piWNT_p)
        NKWPCNT[:] = LHS_WPC - RHS_WPC
