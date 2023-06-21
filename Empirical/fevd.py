import numpy as np
import pandas as pd
import utils
from tqdm import tqdm

def sim_data(estimates, depvar='Y', truth=False, N=32, max_h=20, pp=2, pp_VD=2, true_fact=5):
    dfss = []
    countries = [country for country in estimates if isinstance(estimates[country], dict)]
    for i, country in enumerate(countries):
        A = estimates[country]['A']
        U = estimates[country]['U']
        Z = estimates[country]['Z']
        
        K = U.shape[0]
        T = U.shape[1]
        if truth:
            Z = np.concatenate([estimates[country]['Z']]*true_fact, axis=1)
            T = Z.shape[1]
        U_draw = np.full((K,T), np.nan)
        for i in range(U.shape[0]):
            U_draw[i,:] = np.random.choice(U[i,:], size=T)
        
        Y = A@Z + U_draw
        
        T = Y.shape[1]
        
        dfs = Y
        country_data = [country]*T
        
        dfs = pd.DataFrame(dfs.T)
        dfs.columns = [depvar,'Y_star','P_star','RR_star']
        dfs['country'] = country_data
        
        for t in range(pp+max_h-1):
            dfs[f'{depvar}_{t+1}'] = dfs[depvar].shift(t+1)
            dfs[f'Y_{t+1}_star'] = dfs['Y_star'].shift(t+1)
            dfs[f'P_{t+1}_star'] = dfs['P_star'].shift(t+1)
            dfs[f'RR_{t+1}_star'] = dfs['RR_star'].shift(t+1)
        
        dfss.append(dfs)
            
    dfs = pd.concat(dfss)
    
    dfs = utils.add_country_Ys_dummies(dfs, max_h, max(pp,pp_VD), tradevar='Y')
    dfs = utils.add_country_Ys_dummies(dfs, max_h, max(pp,pp_VD), tradevar='P')
    dfs = utils.add_country_Ys_dummies(dfs, max_h, max(pp,pp_VD), tradevar='RR')
        
    return dfs

def bootstrap_FEVD(depvar='Y', hs=[4,8,20], pp_VD=2, pp=2, prob=0.05, trend_filter='Trend', TFE=True,
                   do_truth=True, true_fact=7, Nt=10, Nb=70):
    # Create panel data set
    var_mask = list(set(['Y'] + [depvar]))
    if depvar == 'NX':
        var_mask += ['EX','IM']
    max_h = max(hs) + 1
    
    # Selection of countries needed to avoid multicollinearity
    df = utils.create_reg_data(max_h, pp=max(pp,pp_VD), trend_filter=trend_filter, var_mask=var_mask, Ys=['Y','P','RR'],
                               quiet=True, add_interactions=True, country_mask=None)

    # Compute FEVD
    FEVD = do_FEVD(df, [depvar], pp=pp_VD, cov_type='standard', hs=hs,
                   TFE=TFE, country_dummies=True, adjusted=False)
    
    # Prepare data
    dfr = df[[f'{depvar}',f'{depvar}_1',f'{depvar}_2',f'{depvar}_3',f'{depvar}_4',
               'Y_star','Y_1_star','Y_2_star','Y_3_star','Y_4_star',
               'P_star','P_1_star','P_2_star','P_3_star','P_4_star',
               'RR_star','RR_1_star','RR_2_star','RR_3_star','RR_4_star',
               'country']].dropna()
    Y = dfr[[depvar,'Y_star','P_star','RR_star']]
    Z = pd.DataFrame(index=Y.index)
    Z['cons'] = 1.0
    
    # Add FE
    FE = pd.get_dummies(Z.index)
    FE.index = Z.index
    Z = pd.concat([Z, FE], axis=1)
    
    # De-FE data
    Y = np.array(Y).T
    Z = np.array(Z).T
    A = Y@Z.T@np.linalg.inv(Z@Z.T)
    U = Y - A@Z
    
    # Add de-FE'ed data
    dfn = pd.DataFrame(index=dfr.index, columns=[depvar,'Y_star','P_star','RR_star'])
    dfn[[depvar,'Y_star','P_star','RR_star']] = U.T
    dfn['country'] = dfr['country']
    
    # Get estimates
    estimates = {}
    countries = list(set(dfn['country']))
    for country in countries:
        # Select country data
        dfc = dfn[dfn['country']==country]
        dfc = dfc[[depvar,'Y_star','P_star','RR_star']].dropna()
        for p in range(4):
            dfc[f'{depvar}_{p+1}'] = dfc[depvar].shift(p+1)
            dfc[f'Y_{p+1}_star'] = dfc['Y_star'].shift(p+1)
            dfc[f'P_{p+1}_star'] = dfc['P_star'].shift(p+1)
            dfc[f'RR_{p+1}_star'] = dfc['RR_star'].shift(p+1)
        dfc = dfc.dropna()
        
        if len(dfc) > 0:
            # Create arrays for OLS
            Y = dfc[[depvar,'Y_star','P_star','RR_star']]
            Z = dfc[[f'{depvar}_1','Y_1_star','P_1_star','RR_1_star',
                     f'{depvar}_2','Y_2_star','P_2_star','RR_2_star',
                     f'{depvar}_3','Y_3_star','P_3_star','RR_3_star',
                     f'{depvar}_4','Y_4_star','P_4_star','RR_4_star']]
            Z['cons'] = 1
            Y = np.array(Y).T
            Z = np.array(Z).T
    
            # Predict data
            A = Y@Z.T@np.linalg.inv(Z@Z.T)
            U = Y - A@Z
            
            estimate = {'A': A, 'U': U, 'Z': Z}
            estimates[country] = estimate
        else:
            estimates[country] = np.nan
            
    if do_truth:
        true_FEVDs = np.full((len(hs), Nt), np.nan)
        for i in tqdm(range(Nt)):
            dfr = sim_data(estimates, depvar, True, 123, max_h, pp, pp_VD, true_fact)

            # Compute FEVD
            FEVD_truth = do_FEVD(dfr, [depvar], pp=pp_VD, cov_type='standard', hs=hs,
                                 TFE=TFE, country_dummies=True, adjusted=False)
            true_FEVDs[:,i] = np.array(FEVD_truth).reshape(-1)
    
    # Bootstrap
    FEVDs = np.full((len(hs), Nb), np.nan)
    for i in tqdm(range(Nb)):
        # Simulate
        dfr = sim_data(estimates, depvar, False, 123, max_h, pp, pp_VD, 1)

        # Compute FEVD
        FEVDr = do_FEVD(dfr, [depvar], pp=pp_VD, cov_type='standard', hs=hs,
                        TFE=TFE, country_dummies=True, adjusted=False)
        FEVDs[:,i] = np.array(FEVDr).reshape(-1)
        
    # Create dataframe
    out = pd.DataFrame(np.percentile(FEVDs, [100*prob/2, 100*(1-prob/2)], axis=1).T, columns=['B_lo','B_hi'], index=hs)
    out['Actual'] = FEVD
    out['B_mean'] = np.mean(FEVDs, axis=1)
    if do_truth:
        out['Truth'] = pd.DataFrame(np.mean(true_FEVDs, axis=1), index=hs)
        out['Bias'] = out['B_mean'] - out['Truth']
        out['Corr'] = out['Actual'] - out['Bias']
        out['B_lo_diff'] = out['B_mean'] - out['B_lo']
        out['B_hi_diff'] = out['B_hi'] - out['B_mean']
        out['Corr_lo'] = out['Corr'] - out['B_lo_diff']
        out['Corr_hi'] = out['Corr'] + out['B_hi_diff']

    # Return stuff
    return out

def do_FEVD(df, dep_vars, pp=4, cov_type='standard', hs=[4,8,20], TFE=True,
            country_dummies=True, adjusted=False):
    FEVD = pd.DataFrame(columns=list(dep_vars), index=hs, dtype=np.float64)
    for depvar in dep_vars:
        for j,h in enumerate(hs):
            controls = ['Y_star', depvar]
            
            # Reduced regresnon
            res_reduced, comps = utils.lp_reg(df, controls, pp=pp, cov_type=cov_type, depvar=depvar,
                                        h=h, TFE=TFE, mode='not', country_dummies=country_dummies)
            ssr_reduced = res_reduced.ssr
            df_reduced = res_reduced.df_resid
    
            # Full regression
            res_full, comps = utils.lp_reg(df, controls, pp=pp, cov_type=cov_type, depvar=depvar,
                                     h=h, TFE=TFE, mode='full', country_dummies=country_dummies)
            ssr_full = res_full.ssr
            df_full = res_full.df_resid
            
            # Compute VD
            if adjusted:
                svd = (1 - (ssr_full/df_full)/(ssr_reduced/df_reduced))*100
            else:
                svd = (1 - ssr_full/ssr_reduced)*100
            FEVD[depvar].iloc[j] = svd
            
    return FEVD