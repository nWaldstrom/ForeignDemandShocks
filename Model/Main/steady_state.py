import numpy as np
import utils
import helpers
from scipy import optimize
from household_problem import compute_RA_jacs


def calibrate_ss(model, do_print=False):
    """ find the steady state """

    par = model.par
    ss = model.ss

    # Prices normalized to 1
    P_list = ['E', 'PNT', 'PH', 'PF_s', 'PT', 'PXT', 'PXNT', 'PH']
    for i in P_list:
        setattr(ss, i, 1.0)

    # Inflation rates are zero in the steady state
    pi_list = ['piNT', 'pi', 'piF', 'piH', 'ppi', 'piF_s', 'piH_s', 'piWT', 'piWNT']
    for i in pi_list:
        setattr(ss, i, 0.0)

    # Use model relations from here on
    ss.PH_s, ss.PF = ss.PH / ss.E, ss.PF_s * ss.E
    ss.PTP = ss.PT / ss.P
    ss.P = helpers.Price_index(ss.PNT, ss.PT, par.etaT, par.alphaT)
    ss.Q = ss.E / ss.P

    # Normalize aggregate GDP and labor to 1
    ss.GDP = 1.0
    ss.N = 1.0

    # Interest rates are all equal to foreign, exogenous rate
    ss.i = ss.iF_s = ss.iF_s = ss.rF_s = par.iF_s_exo
    # Real rates are equal to nominal rates due to zero inflation
    ss.r = ss.ra = ss.i

    # shocks
    ss.C_s = 1.0
    ss.eps_beta = 1.0
    ss.di = 0.0
    par.TFP = 1.0

    # Public sector   
    ss.G_exo = ss.G_trans = 0.0
    ss.UniformT = ss.UniformT_exo = 0.0
    ss.VAT = 0.0
    ss.subP = 0.0
    ss.FD = 0.0

    # Preliminaries 
    ss.mcT = 1 / par.mu[0]
    ss.mcNT = 1 / par.mu[1]
    ss.PH_s = ss.PH / ss.E * (1 - ss.VAT)
    ss.PF = ss.E * ss.PF_s / (1 - ss.VAT)
    ss.ToT = ss.PF / ss.PH_s / ss.E * (1 - ss.VAT)
    ss.ND = ss.E

    # define steady state residuals
    def ss_res(x):
        par.FixedCost[0], par.FixedCost[1], ss.ZNT, ss.ZT, par.X_share_T[2] = x[0], x[1], x[2], x[3], x[4]

        par.X_share_T[0] = (1 - par.X_share_T[2]) * 0.6  # 60% split
        par.X_share_T[1] = (1 - par.X_share_T[2]) * (1 - 0.6)

        ss.NT = ss.N * par.sT
        ss.NNT = ss.N * (1 - par.sT)

        # Intermediate good FOCs 
        exp = (par.sigma_NX - 1) / par.sigma_NX
        if par.X_expshare[0] > 0.0 and par.X_expshare[1] > 0.0:
            def Intermediate_goods_sys(x):

                par.alphaX[0], par.alphaX[1], par.TFP_s[0], par.TFP_s[1] = x

                exp = (par.sigma_NX - 1) / par.sigma_NX
                NT_eff = ss.NT
                NNT_eff = ss.NNT

                beta_T = (1 - par.alphaX[0]) * (ss.ZT / (NT_eff * par.TFP * par.TFP_s[0])) ** exp
                beta_NT = (1 - par.alphaX[1]) * (ss.ZNT / (NNT_eff * par.TFP * par.TFP_s[1])) ** exp

                ss.WT = (1 - ss.VAT) * ss.mcT * beta_T * ss.PH * (par.TFP * par.TFP_s[0]) ** exp * (ss.ZT / NT_eff) ** (
                            1 / par.sigma_NX) / (1 - ss.subP)
                ss.WNT = (1 - ss.VAT) * ss.mcNT * beta_NT * ss.PNT * (par.TFP * par.TFP_s[1]) ** exp * (
                            ss.ZNT / NNT_eff) ** (1 / par.sigma_NX) / (1 - ss.subP)

                ss.XT = par.X_expshare[0] * ss.NT * ss.WT * (1 - ss.subP) / (ss.PXT * (1 - par.X_expshare[0]))
                ss.XNT = par.X_expshare[1] * ss.NNT * ss.WNT * (1 - ss.subP) / (ss.PXNT * (1 - par.X_expshare[1]))

                alpha_T = par.alphaX[0] * (ss.ZT / (ss.XT * par.TFP * par.TFP_s[0])) ** exp
                alpha_NT = par.alphaX[1] * (ss.ZNT / (ss.XNT * par.TFP * par.TFP_s[1])) ** exp

                res1 = ss.XT - alpha_T ** (par.sigma_NX) * (ss.PXT / ((1 - ss.VAT) * ss.PH * ss.mcT)) ** (
                    -par.sigma_NX) * ss.ZT * (par.TFP * par.TFP_s[0]) ** (par.sigma_NX - 1)
                res2 = ss.XNT - alpha_NT ** (par.sigma_NX) * (ss.PXNT / ((1 - ss.VAT) * ss.PNT * ss.mcNT)) ** (
                    -par.sigma_NX) * ss.ZNT * (par.TFP * par.TFP_s[1]) ** (par.sigma_NX - 1)

                res3 = ss.ZT - (par.TFP * par.TFP_s[0] * NT_eff ** (1 - par.alphaX[0]) * ss.XT ** par.alphaX[0])
                res4 = ss.ZNT - (par.TFP * par.TFP_s[1] * NNT_eff ** (1 - par.alphaX[1]) * ss.XNT ** par.alphaX[1])

                return [res1, res2, res3, res4]

            solution = optimize.root(Intermediate_goods_sys, [0.5, 0.5, 1.0, 1.0], method='lm', options={'ftol': 1e-08})

            if not solution.success:
                raise ValueError('Could not solve intermediate goods system')
            residuals = Intermediate_goods_sys(solution.x)

            # check that solution is correct 
            for k in residuals:
                assert abs(k) < 1e-06

            # get solution and update parameters
            par.alphaX[0], par.alphaX[1], par.TFP_s[0], par.TFP_s[1] = solution.x
            par.prodalpha[0] = par.alphaX[0] * (ss.ZT / (ss.XT * par.TFP * par.TFP_s[0])) ** exp
            par.prodalpha[1] = par.alphaX[1] * (ss.ZNT / (ss.XNT * par.TFP * par.TFP_s[1])) ** exp
            par.prodbeta[0] = (1 - par.alphaX[0]) * (ss.ZT / (ss.NT * par.TFP * par.TFP_s[0])) ** exp
            par.prodbeta[1] = (1 - par.alphaX[1]) * (ss.ZNT / (ss.NNT * par.TFP * par.TFP_s[1])) ** exp
            if not np.isclose(par.sigma_NX,
                              1.0):  # If we do not have Cobb-Douglass, check that CES is calibrated to match CD form in steady state
                assert np.isclose(ss.ZT, par.TFP * par.TFP_s[0] * (
                            par.prodbeta[0] * ss.NT ** exp + par.prodalpha[0] * ss.XT ** exp) ** (1 / exp))
                assert np.isclose(ss.ZNT, par.TFP * par.TFP_s[1] * (
                            par.prodbeta[1] * ss.NNT ** exp + par.prodalpha[1] * ss.XNT ** exp) ** (1 / exp))

        # If we do not have intermediate goods (X_expshare=0) but only labor in production
        else:
            ss.XT = ss.XNT = 0.0
            par.alphaX[:] = 0.0
            NT_eff = ss.NT
            NNT_eff = ss.NNT
            par.TFP_s[0] = ss.ZT / (par.TFP * NT_eff)
            par.TFP_s[1] = ss.ZNT / (par.TFP * NNT_eff)
            ss.WT = (1 - ss.VAT) * ss.mcT * ss.PH * (par.TFP * par.TFP_s[0]) ** exp * (ss.ZT / NT_eff) ** (
                        1 / par.sigma_NX) / (1 - ss.subP)
            ss.WNT = (1 - ss.VAT) * ss.mcNT * ss.PNT * (par.TFP * par.TFP_s[1]) ** exp * (ss.ZNT / NNT_eff) ** (
                        1 / par.sigma_NX) / (1 - ss.subP)

        # Get demand for intermediate goods from different sectors using CES formulas
        ss.XT2NT, ss.XNT2T, ss.XNT2NT, ss.XT2T, ss.XM2T = helpers.get_intermediates(ss.XT, ss.XNT, ss.PXT, ss.PXNT,
                                                                                    ss.PH, ss.PNT, ss.PF, par.X_share_T,
                                                                                    par.X_share_NT, par.etaX)

        # Nominal value added is output net intermediate goods and fixed cost
        nOT = ss.PH * ss.ZT - ss.PXT * ss.XT - ss.PH * par.FixedCost[0]
        nONT = ss.PNT * ss.ZNT - ss.PXNT * ss.XNT - ss.PNT * par.FixedCost[1]
        ss.PGDP = (nOT + nONT) / ss.GDP  # GDP deflator is nominal GDP over real GDP
        # Real value added is nominal value added deflated with the GDP deflator
        ss.YT = nOT / ss.PGDP
        ss.YNT = nONT / ss.PGDP

        # Other aggregates and accounting
        ss.Y = ss.YT + ss.YNT
        ss.Z = (ss.PH * ss.ZT + ss.PNT * ss.ZNT) / ss.PGDP
        ss.PO_T = nOT / ss.GDP_T
        ss.PO_NT = nONT / ss.GDP_NT
        ss.OT = ss.YT * ss.PGDP / ss.PO_T
        ss.ONT = ss.YNT * ss.PGDP / ss.PO_NT

        # Wages
        ss.wnT = (ss.WT * ss.NT / ss.P) / par.sT
        ss.wnNT = ((ss.WNT * ss.NNT / ss.P) / (1 - par.sT))

        ss.wT, ss.wNT = ss.wnT * par.sT / ss.NT, ss.wnNT * (1 - par.sT) / ss.NNT
        ss.w = ss.wT + ss.wNT

        # Dividends and asset pricing
        ss.DivT = ((1 - ss.VAT) * nOT - ss.WT * ss.NT * (1 - ss.subP)) / ss.P
        ss.DivNT = ((1 - ss.VAT) * nONT - ss.WNT * ss.NNT * (1 - ss.subP)) / ss.P
        ss.Div = ss.DivT + ss.DivNT
        ss.pD = ss.Div / ss.r

        # Public sector (bonds, taxes, spending)
        ss.G = par.G_GDP_ratio * ss.GDP * ss.PGDP / ss.P
        ss.B = par.B_GDP_ratio * ss.GDP * ss.PGDP / ss.P
        Tax_revenue = ss.VAT * (nOT + nONT) / ss.P - ss.subP * (ss.WT * ss.NT + ss.WNT * ss.NNT) / ss.P
        ss.LT = ss.r * ss.B + ss.G - Tax_revenue

        # Households
        ss.Income = ss.wnNT * (1 - par.sT) + ss.wnT * par.sT - ss.LT + ss.UniformT + ss.UniformT_exo
        A = ss.pD + ss.B  # or A = par.W2INC_target * ss.Income
        ss.C = ss.r * A + ss.Income

        # Aggregate trade flows
        # Exports
        nExports = par.X2Y_target * ss.GDP
        ss.Exports = nExports * ss.PGDP / ss.P

        # Using that imports = exports in steady state (NFA = 0 by assumption)
        ss.Imports = ss.Exports
        HH_imports = ss.Imports - ss.XM2T * ss.PF / ss.P
        # Back out alpha that is consistent with implied level of HH imports
        par.alpha = HH_imports / (par.alphaT * ss.C)

        # Get consumption of various goods from CES
        ss.CT, ss.CNT = helpers.CES_demand(ss.C, ss.PNT, ss.PT, ss.P, par.etaT, par.alphaT)
        ss.CF, ss.CH = helpers.CES_demand(ss.CT, ss.PH, ss.PF, ss.PT, par.eta, par.alpha)
        # Back out alphaF such that foreign demand = exports
        par.alpha_F = ss.Exports * ss.PGDP / ss.P
        ss.CH_s = helpers.Armington(ss.PH_s, ss.PF_s, ss.C_s, par.gamma, par.alpha_F)

        # Goods market clearing
        ss.GDP_T = ss.ZT - ss.XT2NT - ss.XT2T - par.FixedCost[0]
        ss.GDP_NT = ss.ZNT - ss.XNT2T - ss.XNT2NT - par.FixedCost[1]
        ss.goods_mkt_T = ss.GDP_T - (ss.CH + ss.CH_s + ss.G * par.sGT_ss)
        ss.goods_mkt_NT = ss.GDP_NT - (ss.CNT + ss.G * (1 - par.sGT_ss))

        # Calibration targets
        Asset_target = (ss.pD + ss.B) / ss.Income - par.W2INC_target  # wealth-income ratio
        profit_target = ss.DivT / ss.Div - ss.GDP_T / ss.GDP  # Dividend share in sectors should be same as GDP as share

        if par.X_expshare[0] > 0.0 and par.X_expshare[1] > 0.0:
            Import_Target = HH_imports / ss.Imports - par.HH_importshare
        else:
            Import_Target = par.X_share_T[2] - 0.0

        return np.array([Asset_target, ss.goods_mkt_NT, ss.goods_mkt_T, profit_target, Import_Target])

    # Calibrate steady state
    if do_print: print('Solving calibration:')
    results = utils.broyden_solver_autojac(ss_res, x0=par.x0, maxcount=150, noisy=do_print, tol=1E-10)
    ss_res(results[0])  # evaluation at solution
    par.x0 = results[0]
    par.FixedCost[0], par.FixedCost[1], ss.ZNT, ss.ZT, par.X_share_T[2] = results[0]
    par.X_share_T[0] = (1 - par.X_share_T[2]) * 0.6
    par.X_share_T[1] = (1 - par.X_share_T[2]) * (1 - 0.6)
    assert np.isclose(sum(par.X_share_T), 1.0)

    #########################
    #   Household problem   #
    #########################

    if par.HH_type == 'HA':
        def HH_res(x):
            par.beta_mean, par.beta_delta = x[0], x[1]

            maxbeta = max(par.beta_grid)
            if maxbeta >= 0.998:
                print('Max beta reached!')
                maxbeta = 0.9998 * 1 / (1 + ss.r) - 0.0005
                par.beta_delta = maxbeta - par.beta_mean

            model.solve_hh_ss(do_print=False)
            model.simulate_hh_ss(do_print=False)

            ss.A = np.sum(ss.D * ss.a)
            ss.C = np.sum(ss.D * ss.c)

            Target_first_year_MPC = True

            MPC_ann, MPC_quarterly, _, _ = utils.nonlin_MPC(model)
            if Target_first_year_MPC:
                MPC_res = MPC_ann[0] - par.Agg_MPC
            else:
                MPC_res = MPC_quarterly[0] - 0.19961746

            return np.array([(ss.A - ss.pD - ss.B), MPC_res])

        if do_print: print('Solving HA calibration:')
        results_HH = utils.broyden_solver_autojac(HH_res, x0=par.x0_het, maxcount=30, noisy=do_print, tol=1E-11)
        par.x0_het = results_HH[0]
        HH_res(results_HH[0])

        if par.HA_PooledInc:
            ss.UCT = ss.C ** (-par.CRRA)
            ss.UCNT = ss.C ** (-par.CRRA)
        else:
            ss.A = np.sum(ss.D * ss.a)
            ss.C = np.sum(ss.D * ss.c)
            ss.C_T = np.sum(ss.D * ss.c_T)
            ss.C_NT = np.sum(ss.D * ss.c_NT)
            ss.C_T_hh = np.sum(ss.D * ss.c_T)
            ss.C_NT_hh = np.sum(ss.D * ss.c_NT)
            ss.UC_T = ss.C_T ** (-par.CRRA)
            ss.UC_NT = ss.C_NT ** (-par.CRRA)

            ss.C_T = np.sum(ss.D * ss.c_T)
            ss.C_T = np.sum(ss.D * ss.c_T)

            ss.C_hh = ss.C
            ss.A_hh = ss.A

            ss.INC_T_hh = np.sum(ss.D * ss.inc_T)
            ss.INC_NT_hh = np.sum(ss.D * ss.inc_NT)


    elif par.HH_type in par.No_HA_list:
        ss.A = ss.pD + ss.B
        par.beta_mean = 1 / (1 + ss.r)
        ss.UC_T, ss.UC_NT = ss.C ** (-par.CRRA), ss.C ** (-par.CRRA)
        ss.C_T = ss.C_NT = ss.C
        if par.HH_type == 'TA-CM' or par.HH_type == 'TA-IM':
            ss.CHtM = ss.Income
            ss.CR = ss.r * ss.A / (1 - par.Agg_MPC) + ss.Income
        else:
            ss.CR = ss.C
            ss.CHtM = 0.0
        compute_RA_jacs(ss, par)
    else:
        raise ValueError('Incorrect HH type chosen!')

    # Back out disutility of work psi such that NKPWC holds in steady state
    par.psi[0] = ss.wT * (par.epsilon_w - 1) / par.epsilon_w * ss.UC_T / (ss.NT / par.sT) ** par.inv_frisch
    par.psi[1] = ss.wNT * (par.epsilon_w - 1) / par.epsilon_w * ss.UC_NT / (ss.NNT / (1 - par.sT)) ** par.inv_frisch

    # Final evaluation at calibrated parameters
    if par.HH_type == 'HA':
        model.solve_hh_ss(do_print=False)
        model.simulate_hh_ss(do_print=False)

    # misc
    par.theta[:] = par.epsilon / par.NKslope
    ss.NFA = ss.A - ss.pD - ss.B
    ss.goods_mkt = ss.Z * ss.PGDP - ss.XT2NT - ss.XNT2T - ss.XNT2NT - ss.XT2T - par.FixedCost[0] - par.FixedCost[1] - (
                ss.CH_s + ss.CNT + ss.CH + ss.G)
    ss.goods_mkt_T = ss.ZT - ss.XT2NT - ss.XT2T - par.FixedCost[0] - (ss.CH + ss.CH_s + ss.G * par.sGT_ss)
    ss.goods_mkt_NT = ss.ZNT - ss.XNT2T - ss.XNT2NT - par.FixedCost[1] - (ss.CNT + ss.G * (1 - par.sGT_ss))

    assert np.isclose(ss.Z * ss.PGDP, ss.ZT * ss.PH + ss.ZNT * ss.PNT)

    ss.NX = (ss.PH * ss.ZT + ss.PNT * ss.ZNT) / ss.P - (
                ss.C + ss.G + ss.XT2NT + ss.XNT2T + ss.XNT2NT + ss.XT2T + ss.XM2T) - par.FixedCost[0] - par.FixedCost[
                1]  # + r_pay
    ss.Exports = ss.CH_s * ss.PH_s / ss.P
    ss.Imports = (ss.CF + ss.XM2T) * ss.PF / ss.P
    assert np.isclose(ss.Exports - ss.Imports, ss.NX, atol=1e-07)

    assert abs(ss.C - (ss.WNT * ss.NNT + ss.WT * ss.NT + ss.r * ss.A - ss.LT)) < 1e-07
    ss.Walras = ss.NX + ss.r * ss.NFA
    ss.DomP = (ss.PNT * ss.CNT + ss.PH * ss.CH) / (ss.PNT * ss.CNT + ss.PH * ss.CH)

    # Update targets and endo. inputs to model type 
    model.HH_type(par.HH_type)
    model.use_FD_shock(par.FD_shock)

    # update shock persistence and standard deviation if changed 
    for var in model.shocks:
        setattr(par, 'jump_' + var, 0.01)
        setattr(par, 'rho_' + var, par.rho)

    if do_print:
        print(f'Implied G/GDP = {ss.G * ss.PGDP / (ss.GDP * ss.PGDP):6.3f}')
        print(f'Implied Exports/GDP = {ss.Exports * ss.PGDP / (ss.GDP * ss.PGDP):6.3f}')
        print(f'Implied Imports/GDP = {ss.Imports * ss.PGDP / (ss.GDP * ss.PGDP):6.3f}')
        print(f'Implied A/GDP = {ss.A * ss.P / (ss.GDP * ss.PGDP):6.3f}')
        print(f'Implied NX/GDP = {ss.NX / (ss.PGDP * ss.GDP):6.3f}')
        print(f'Implied C/GDP = {ss.C * ss.P / (ss.PGDP * ss.GDP):6.3f}')
        print(f'Goods market eq. residual = {ss.goods_mkt:12.8f}')
        print(f'Goods market NT = {ss.goods_mkt_NT:12.8f}')
        print(f'Goods market T = {ss.goods_mkt_T:12.8f}')
        print(f'Walras = {ss.Walras:12.8f}')
