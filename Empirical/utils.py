import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import oeap
import math
import itertools
import filtering
from scipy.stats import norm
from statsmodels.tools.tools import add_constant
from statsmodels.regression.linear_model import OLS
from seaborn import set_palette
from matplotlib import rc
from matplotlib.ticker import FormatStrFormatter

# Supress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
pd.options.mode.chained_assignment = None

# Plot style
plt.style.use('seaborn-white')
set_palette("colorblind")
rc('font', **{'family': 'serif', 'serif': ['Palatino']})
rc('text', usetex=True)

def filter_for_panel(trend_filter='Trend', var_mask=None, country_mask=None,
                     sample_start=None, sample_end=None, large=False,
                     exch=True, infl=True, R=True):
    # Load data
    # Note: All columns must have format '{var}_{country}'
    if large:
        df = pd.read_pickle('Data/oecd_large.pkl')
    else:
        df = pd.read_pickle('Data/oecd.pkl')
    countries = set([col.split('_')[1] for col in df])
      
    if exch:
        df_exch = pd.read_excel('Data/ExchRate.xlsx', skiprows=53, index_col=0,
                                usecols='A:GS').transpose()
        T_df_exch = len(df_exch)
        df_exch = df_exch[[col for col in df_exch if col in countries]]
        df_exch.columns = [f'E_{col}' for col in df_exch]
        df_exch.reset_index(inplace=True, drop=True)
        arr_exch = np.zeros((len(df),len(df_exch.columns))) + np.nan
        arr_exch[-T_df_exch:] = np.array(df_exch)
        df_exch = pd.DataFrame(arr_exch, columns=df_exch.columns,
                               index=df.index)
        df = pd.concat([df, df_exch], axis=1)
        
    if infl:
        df_P = (pd.read_excel('Data/CPI.xlsx', index_col=0)
                  .replace('..', np.nan))
        df_CPI = df_P/df_P.shift(4)-1
        df_CPI = np.exp(df_CPI)
        df = pd.concat([df, df_CPI], axis=1)
        
        df_CPI_shift = df_CPI.shift(-4)
        df_CPI_shift.columns = [col.split('_')[0] + 'I_' + col.split('_')[1] 
                            for col in df_CPI_shift.columns]
        df = pd.concat([df, df_CPI_shift], axis=1)
    
    if R:
        df_R = (pd.read_excel('Data/Rshort.xlsx', index_col=0)
                .replace('..', np.nan)/100)
        df_R = np.exp(df_R)
        df = pd.concat([df, df_R], axis=1)
        
        for col in df:
            country = col.split('_')[1]
            var = col.split('_')[0]
            if var == 'R':
                if f'PI_{country}' in df:
                    df[f'RR_{country}'] = np.exp((1+np.log(df[f'R_{country}']))/(1+np.log(df[f'PI_{country}']))-1)
                else:
                    df[f'RR_{country}'] = np.nan
    
    # Choose sample
    if sample_start is None:
        sample_start = df.index[0]
    if sample_end is None:
        sample_end = df.index[-1]
    df = df[sample_start:sample_end]
    
    # Select only relevant variables and countries
    selected_cols = df.columns
    if var_mask is not None:
        selected_cols = [col for col in selected_cols if col.split('_')[0]
                         in var_mask]
    df = df[selected_cols]
    
    selected_cols = df.columns
    if country_mask is not None:
        selected_cols = [col for col in selected_cols if col.split('_')[1]
                         in country_mask]
    df = df[selected_cols]
        
    # Filtering
    for col in df:
        if len(df[col].dropna()) > 0 and (df[col].dropna()>0).all() > 0:
            country = col.split('_')[1]
            filtered = filtering.filter_data(np.log(df[col]), trend_filter)[0].dropna()
            filtered.columns = range(filtered.shape[1])
            df[col] = filtered
        else:
            df[col] = np.nan
    
    return df

def create_panel(lags=4, trend_filter='Trend', h=None, var_mask=None,
                 country_mask=None, sample_start=None, sample_end=None,
                 large=False, exch=True, infl='CPI', R=True):
    # Get filtered data
    df = filter_for_panel(trend_filter=trend_filter, var_mask=var_mask,
                          country_mask=country_mask, sample_start=sample_start,
                          sample_end=sample_end, large=large,
                          exch=exch, infl=infl, R=R)
            
    # Create auxiliary variables
    countries = list(set([col.split('_')[1] for col in df]))
    if 'NX' in var_mask:
        for country in countries:
            if f'EX_{country}' in df and f'IM_{country}' in df:
                df[f'NX_{country}'] = df[f'EX_{country}'] - df[f'IM_{country}']
            else:
                df[f'NX_{country}'] = np.nan
                
    # Sum over horizon
    if h is not None:
        for col in df:
            var, country = col.split('_')
            df[f'{var}cum_{country}'] = df[col].rolling(h+1).sum().shift(-h)
                
    # Store stuff
    varlist = list(set([col.split('_')[0] for col in df]))
    N = len(countries)
    T = len(df)
    
    # Create lags
    for col in df:
        var, country = col.split('_')
        for lag in range(lags+1):
            df[f'{var}_{lag}_{country}'] = df[f'{var}_{country}'].shift(lag)
        df = df.drop(col, axis=1)
        
    # Put all variables, all countries, and all lags into panel
    df_panel = pd.DataFrame(index=np.tile(df.index, N))
    for var in varlist:
        for lag in range(lags+1):
            selected_cols = []
            for country in countries:
                col = f'{var}_{lag}_{country}'
                if col not in df:
                    df[col] = np.nan
                selected_cols.append(col)
            var_flat = np.array(df[selected_cols]).flatten(order='F')
            if lag == 0:
                df_panel[var] = var_flat
            else:
                df_panel[f'{var}_{lag}'] = var_flat
            
    # Create numeric time variables
    df_panel['c'] = np.ones(N*T)
    df_panel['t'] = np.tile(np.arange(T), N)
    df_panel['t2'] = np.tile(np.arange(T)**2, N)
    df_panel['t3'] = np.tile(np.arange(T)**3, N)
    
    # Create country variable
    df_panel['country'] = np.repeat(countries, T)
    
    return df_panel

def get_panel_info(df):
    # Full dataset (without dropping observations)
    N = len(set(df['country']))
    T = len(df)
    lags = max([int(col.split('_')[1]) for col in df if
                len(col.split('_')) > 1 if (col.split('_')[1]).isnumeric()])
    print('- FULL DATA INFO -')
    print(f'Number of countries: {N}')
    print(f'Number of columns: {len(df.columns)}')
    print(f'Sample size: {T}')
    print(f'Average sample size per country: {T/N:.0f}')
    print(f'Max lag: {lags:.0f}')
    print(f'Effective sample start: {df.index.min()}')
    print(f'Effective sample end: {df.index.max()}')
    print(' ')
    
    # Dataset focusing on Y (dropping observations)
    dfy = df[['Y','Y_star','country']].dropna()
    N = len(set(dfy['country']))
    T = len(dfy)
    print(' - Y DATA INFO - ')
    print(f'Number of countries: {N}')
    print(f'Number of columns: {len(dfy.columns)}')
    print(f'Sample size: {T}')
    print(f'Average sample size per country: {T/N:.0f}')
    print(f'Effective sample start: {dfy.index.min()}')
    print(f'Effective sample end: {dfy.index.max()}')
    print(' ')
    
def get_oecd_data(large=False):
    # Some options
    dataset = 'QNA'
    locations = ['']
    frequencies = ['Q']
    starttime = '1947-Q1'
    endtime = '2019-Q4'
    if large:
        banned_countries = ['EA19', 'EU27_2020']
    else:
        banned_countries = ['EA19', 'EU27_2020', 'JPN', 'DEU', 'GBR', 'IND',
                            'FRA', 'ITA', 'USA']
        
    # National account aggregates
    vardict = {
        'B1_GE': 'Y',
        'P31S14_S15': 'C',
        'P5': 'I',
        'P6': 'EX',
        'P7': 'IM',
        'P313B': 'CT',
        'P314B': 'CNT',
        'B1GVC': 'YT', # Manufacturing
        'B1GVG_U': 'YNT' # Services
    }
    subjects = list(vardict.keys())
    measures  = ['LNBQRSA'] # Can also use CQRSA for W or HRSSA for N
    dimensions = [locations, subjects, measures, frequencies]
    df1 = oeap.get_df(dataset, dimensions, vardict, banned_countries,
                      starttime, endtime)
    
    # Nominal GDP (for price index)
    vardict = {
        'B1_GE': 'PY'
    }
    subjects = list(vardict.keys())
    measures  = ['CQRSA']
    dimensions = [locations, subjects, measures, frequencies]
    df2 = oeap.get_df(dataset, dimensions, vardict, banned_countries,
                      starttime, endtime)
    
    # Save data to disk
    df = pd.concat([df1, df2], axis=1)
    if large:
        df.to_pickle('Data/oecd_large.pkl')
    else:
        df.to_pickle('Data/oecd.pkl')

def get_trade_data():
    dataset = 'TIVA_2021_C1'
    indicators = ['EXGR']
    industries = ['DTOTAL']
    starttime = '1995'
    endtime = '2018'

    dimensions = [indicators, [''], [''], industries]
    df = oeap.get_df(dataset, dimensions, None, [''], starttime, endtime)
    df.to_pickle('Data/trade.pkl')
    
    return df

def get_endog_var(h, mode='contemp', cum=False, shock='Y_star',
                  fixed_dummy=False, country_dummies=False, countries=None,
                  all_countries=False):
    var = shock.split('_')[0]
    endog_var = []
    if fixed_dummy:
        if h == 0:
            endog_var.append(f'{var}_star_fixed')
        else:
            endog_var.append(f'{var}_{h}_star_fixed')
        
    if mode == 'contemp':
        if not all_countries:
            if h == 0:
                if cum:
                    endog_var += [f'{var}cum']
                else:
                    endog_var += [var]
            else:
                if cum:
                    endog_var += [f'{var}cum_{h}']
                else:
                    endog_var += [f'{var}_{h}']
    else:
        if cum:
            endog_var += [f'{var}cum_{h+1}']
        else:
            endog_var += [f'{var}_{h+1}']
            
    if 'star' in shock and not all_countries:
        endog_var[-1] = endog_var[-1] + '_star'
        
    if country_dummies and mode == 'contemp':
        for country in countries:
            if h == 0:
                country_dummy = f'{shock}_{country}'
            else:
                country_dummy = f'{var}_{h+1}_star_{country}'
            endog_var.append(country_dummy)
    return endog_var

def get_exog_list(h, control_vars, pp, mode, country_dummies,
                  countries, sign):
    exog_list = []
    for var in control_vars:
        if 'star' not in var:
            for lag in range(pp):
                exog_list += [f'{var}_{h+lag+1}']
        else:
            for lag in range(pp):
                vartype = var.split('_')[0]
                exog_list += [f'{vartype}_{h+lag+1}_star']
                
    if mode == 'full':
        exog_list += ['Y_star']
        if sign: exog_list += ['P_star','RR_star']
        if sign:
            for lag in range(h):
                exog_list += [f'Y_{lag+1}_star',f'P_{lag+1}_star',f'RR_{lag+1}_star']
        else:
            for lag in range(h):
                exog_list += [f'Y_{lag+1}_star']
        if country_dummies:
            for country in countries:
                if sign:
                    exog_list += [f'Y_star_{country}',f'P_star_{country}',f'RR_star_{country}']
                    for lag in range(h):
                        exog_list += [f'Y_{lag+1}_star_{country}',f'P_{lag+1}_star_{country}',f'RR_{lag+1}_star_{country}']
                else:
                    exog_list += [f'Y_star_{country}']
                    for lag in range(h):
                        exog_list += [f'Y_{lag+1}_star_{country}']
    
    return exog_list

def get_endog_exog(h, control_vars, pp,  mode='contemp',
                         cum=False, shock='Y_star', fixed_dummy=False,
                         country_dummies=False, df=None, all_countries=False,
                         sign=False):
    if country_dummies:
        if all_countries:
            country_D = pd.get_dummies(df['country'], drop_first=False).columns
        else:
            country_D = pd.get_dummies(df['country'], drop_first=True).columns
        countries = []
        for country in country_D:
            if h==0:
                country_dummy = f'Y_star_{country}'
            else:
                country_dummy = f'Y_{h+1}_star_{country}'
            if np.sum(np.abs(df[country_dummy]) > 1e-8) > 0:
                countries.append(country)
    else:
        countries = None
    
    endog_var = get_endog_var(h, mode, cum, shock, fixed_dummy,
                              country_dummies, countries, all_countries)
    exog_list_init = get_exog_list(h, control_vars, pp, mode,
                                   country_dummies, countries, sign)
    exog_list = [var for var in exog_list_init if var not in endog_var]
    
    return endog_var, exog_list

def lp_reg(df, control_vars, pp=4, cov_type='standard',
           depvar='Y', h=0, TFE=True, min_FE=5, mode='contemp',
           cum=False, shock='Y_star', regions=None, fixed_dummy=False,
           country_dummies=False, warnings=False, const=True,
           all_countries=False, sign=False, do_reg=False, pre_trend_reg=False):
    # Get rid of zero columns
    num_cols = list(df.select_dtypes(include=[np.number]).columns.values)
    other_cols = list(set(df.columns) - set(num_cols))
    non_zero_cols = list((df[num_cols] > 1e-6).any(axis=0).index) + other_cols
    df = df[non_zero_cols]
    
    # Get variables
    endog_var, exog_list = get_endog_exog(h, control_vars, pp, mode, cum,
                                          shock, fixed_dummy, country_dummies,
                                          df, all_countries, sign)
    
    if sign and mode == 'contemp':
        if h == 0:
            endog_var += ['P_star','RR_star']
            if 'star' not in depvar:
                for p in range(pp):
                    exog_list += [f'P_{p+1}_star',f'RR_{p+1}_star']
        else:
            endog_var += [f'P_{h}_star',f'RR_{h}_star']
            if 'star' not in depvar:
                for p in range(pp):
                    exog_list += [f'P_{h+p+1}_star',f'RR_{h+p+1}_star']    
            for var in ['Y','P','RR']:
                for p in range(pp):
                    if f'{var}_{h+p+1}_star' not in exog_list:
                        exog_list.append(f'{var}_{h+p+1}_star')
                        
    if sign and mode in ['full','not']:
        for var in ['P','RR']:
            for p in range(pp):
                exog_list += [f'{var}_{h+p+1}_star']
    
    if shock == depvar and h == 0 and do_reg:
        endog_var = []
    
    if pre_trend_reg:
        depvar = f'Y_{-h}'
        endog_var = [shock]
        var = shock.split('_')[0]
        exog_list = []
        for p in range(pp):
            exog_list += [f'{var}_{-h+p}_star']
    
    sel_vars = [depvar] + endog_var + exog_list
    if regions is not None:
        sel_vars += regions
    sel_vars += ['country']
    
    df = df[sel_vars].dropna()

    dependent = df[depvar]
    endog = df[endog_var]
    exog = df[exog_list]
    countries = df['country']
        
    if TFE:
        FE = pd.get_dummies(df.index)
        FE.index = df.index
        if regions:
            for region in regions:
                RFE = (np.array(FE).T*np.array(df[region])).T
                RFE_cols = [f'{col}_{region}' for col in FE]
                RFE = pd.DataFrame(RFE, columns=RFE_cols)
                FE = pd.DataFrame(np.concatenate([np.array(FE), np.array(RFE)],
                                                 axis=1),
                                  index=FE.index,
                                  columns=list(FE.columns) + list(RFE.columns))
        FE = FE[FE.columns[FE.sum()>=min_FE]]
        
        dependent = dependent[dependent.index.isin(FE.columns)]
        endog = endog[endog.index.isin(FE.columns)]
        exog = exog[exog.index.isin(FE.columns)]
        countries = countries[countries.index.isin(FE.columns)]
        FE = FE[FE.index.isin(FE.columns)]
        FE = FE.iloc[:,1:]
        
        exog = pd.concat([exog, FE], axis=1)
    else:
        exog = df[exog_list]
    if const:
        exog = add_constant(exog)
    
    # Estimate model depending on desired SE's
    if cov_type == 'standard':
        res = OLS(dependent, pd.concat([exog, endog], axis=1)).fit()
    elif cov_type == 'cluster':
        res = (OLS(dependent, pd.concat([exog, endog], axis=1))
               .fit()
               .get_robustcov_results(cov_type='cluster',
                                      groups=exog.index))
    elif cov_type == 'HAC':
        res = (OLS(dependent, pd.concat([exog, endog], axis=1))
               .fit()
               .get_robustcov_results(cov_type='HAC', kernel='bartlett',
                                      maxlags=20))
    elif cov_type == 'DC':
        res = (OLS(dependent, pd.concat([exog, endog], axis=1))
               .fit()
               .get_robustcov_results(cov_type='hac-groupsum', time=exog.index.astype(int), maxlags=3))
    
    if warnings:
        con_no = res.condition_number
        if con_no > 1e+6: print(f'Warning! Condition number = {con_no:.0f}')
    
    comps = {
        'Z': endog,
        'Y': dependent,
        'Q': exog,
        'countries': countries,
        'resid': res.resid,
        'bhat': np.array(res.params)[-1],
        'df': df
    }
    
    return res, comps

def compute_irfs(df, dep_vars, controls, max_h=16, pp=4, cov_type='cluster',
                 min_FE=5, TFE=True, mode='contemp',
                 shock='Y_star', regions=None, fixed_dummy=False,
                 country_dummies=False, sign=False, pre_trend=False):
    if pre_trend:
        h_range = np.arange(-8, max_h)
    else:
        h_range = range(max_h)
    Nh = len(h_range)
    
    N_var = len(dep_vars)
    IRF = np.zeros((N_var, Nh)) + np.nan
    SE = np.zeros((N_var, Nh)) + np.nan
    RES = {}

    for i,var in enumerate(dep_vars):
        for j,h in enumerate(h_range):
            if h < 0:
                pre_trend_reg = True
            else:
                pre_trend_reg = False
            if isinstance(controls, list):
                control_vars = controls
            else:
                if var in controls:
                    if isinstance(controls[var], list):
                        control_vars = controls[var]
                    else:
                        control_vars = controls[controls[var]]
                else:
                    control_vars = [shock, var]
            
            if 'star' in var and 'star' in shock and sign:
                sign_bool = True
            else:
                sign_bool = False
            if (var == shock or sign_bool) and h == 0:
                irf = 1.0
                se = 0.0
                res = None
            else:
                res, _ = lp_reg(df=df, control_vars=control_vars, pp=pp,
                             cov_type=cov_type,
                             depvar=var, h=h, TFE=TFE, min_FE=min_FE,
                             mode=mode, shock=shock, regions=regions,
                             fixed_dummy=fixed_dummy,
                             country_dummies=country_dummies, sign=sign, pre_trend_reg=pre_trend_reg)
                irf = np.array(res.params)[-1]
                se = np.array(res.bse)[-1]

            IRF[i,j] = irf
            SE[i,j] = se
            RES.update({f'{var}_{h}': res})
            
    return IRF, SE, RES
            
def plot_irfs(IRF, SE, dep_vars, prob=0.05, title=None, save=None, betas=None,
              ses=None, max_cols=3, LO=None, HI=None, pre_trend=False):
    sign = norm.ppf(1-prob/2)
    max_h = IRF.shape[1]
    if pre_trend: max_h -= 8
    if SE is not None:
        LO = IRF - sign*SE
        HI = IRF + sign*SE
    if pre_trend:
        x = np.arange(-7,max_h+1)
    else:
        x = range(max_h)
    
    fixed_dummy = betas is not None and ses is not None
    
    N_dep_vars = len(dep_vars)
    n_rows, n_cols = math.ceil(N_dep_vars/max_cols), min(N_dep_vars, max_cols)
    x_size, y_size = 9/max_cols*n_cols, 6/max_cols*n_rows
    fig = plt.figure(figsize=(x_size, y_size), dpi=100)
    fig.subplots_adjust(hspace=0.4)
    fig.subplots_adjust(wspace=0.4)

    for i, var in enumerate(dep_vars):
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        var_name = dep_vars[var]
        ax.plot(x, IRF[i,:], label='Floating')
        if len(LO.shape) == 3:
            for j in range(LO.shape[1]):
                ax.fill_between(x, LO[i,j,:], HI[i,j,:], alpha=0.15, color='C0')
        else:
            ax.fill_between(x, LO[i,:], HI[i,:], alpha=0.15, color='C0')
        if SE is not None:
            ax.fill_between(x, IRF[i,:] - SE[i,:], IRF[i,:] + SE[i,:], alpha=0.08,
                            color='C0')
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.axvline(x=0, color='black', linewidth=0.5, linestyle='--')
        ax.set_xlim(x[0], x[-1]-1)
        ax.set_title(var_name)
        if pre_trend:
            ax.set_xticks(np.arange(-4, max_h+1, 4))
        else:
            ax.set_xticks(np.arange(0, max_h, 4))
        
        if var in ['P','RR','P_star','RR_star']:
            ax.set_ylabel('\%-point diff. to trend')
        else:
            ax.set_ylabel('\% diff. to trend')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        if i >= n_cols*(n_rows-1):
            ax.set_xlabel('Quarters')
        
        if fixed_dummy:
            ax.plot(x, betas[i,:] + IRF[i,:], label='Fixed')
            if i == 0:
                leg = ax.legend(fontsize=8, frameon=True)
                leg.get_frame().set_edgecolor('black')
        
    if title is not None:
        fig.suptitle(title, fontsize=14)
        
    if save is not None:
        plt.savefig(save, bbox_inches='tight')
        
def indicate_region(country, region, df_r):
    return df_r.loc[country, region]
    
def create_reg_data(max_h, pp, trend_filter, var_mask, country_mask=None,
                    sample_start=None, sample_end=None, h=None, drop=False,
                    large=False, quiet=False, Ys=None, regions=False,
                    exch=True, infl='CPI', R=True, years=False,
                    cont_countries=False, add_interactions=False, regime='All',
                    only_big=False, common_weights=False):
    lags = max_h + pp - 1
    df = create_panel(lags=lags, trend_filter=trend_filter, h=h,
                      var_mask=var_mask, country_mask=country_mask, sample_start=sample_start,
                      sample_end=sample_end, large=large, exch=exch,
                      infl=infl, R=R)
    
    if Ys is not None:
        df_w = create_weights(only_big=only_big, common_weights=common_weights)
        for tradevar in Ys:
            df = add_trade_var(df, df_w, lags, years=years, tradevar=tradevar,
                               cont_countries=cont_countries,
                               trend_filter=trend_filter, h=h)
    if regions:
        df_r = pd.read_excel('Data/Countries.xlsx', usecols='A:H', index_col=0)
        regions = list(df_r.columns)
        for region in regions:
            df[region] = df['country'].apply(indicate_region, region=region,
                                             df_r=df_r)
    df = df.dropna(axis=1, how='all')
    if drop:
        df = df.dropna()
    if add_interactions:
        for tradevar in Ys:
            df = add_country_Ys_dummies(df, max_h, pp, tradevar)
    if regime != 'All':
        df_f = pd.read_excel('Data/Fixed.xlsx', usecols='C:BA', skiprows=2, index_col=0)
        df_f = df_f.iloc[1:,:]
        df['Regime'] = np.nan
        for i in range(df.shape[0]):
            time = df.index[i]
            country = df['country'].iloc[i]
            if country in df_f:
                df['Regime'][i] = df_f[country][time]
            else:
                df['Regime'][i] = np.nan
        if regime == 'Fixed':
            df = df[df['Regime'] == 1.0]
        elif regime == 'Floating':
            df = df[df['Regime'] == 0.0]
    if not large:
        df = df[~df['country'].isin(['USA','CHN','JPN','DEU','GBR','IND','FRA','ITA'])]
    if not quiet:
        get_panel_info(df)
    
    return df

def create_weights(from_web=False, year=None,
                   bigs=['USA','CHN','JPN','DEU','GBR','IND','FRA','ITA'],
                   only_big=False, common_weights=False):
    if common_weights:
        df_all = pd.read_excel('Data\Levels.xlsx', index_col=0, header=None)
        df_all.index.name = None
        df_w = pd.DataFrame(index=df_all.index, columns=df_all.index)
        for col in df_w:
            df_w[col] = df_all
    else:
        df = create_reg_data(1, 1, 'Trend', ['Y'], drop=False, quiet=True)
        countries = sorted(list(set(list(set(df['country'])) + bigs)))
        pairs = list(itertools.product(countries,countries))
        cols = [f'{pair[0]}_{pair[1]}' for pair in pairs]
    
        if from_web:
            df = get_trade_data()
        else:
            df = pd.read_pickle('Data/trade.pkl')
        # Naming {import from}_{importer} / {export_to}_{exporter}
    
        df_out = pd.DataFrame(index=df.index)
        for col in cols:
            if col in df:
                df_out[col] = df[col]
        df = df_out
        
        df_w = pd.DataFrame(columns=countries, index=countries)
        for importer in countries:
            cols = [col for col in df if col.split('_')[1] == importer]
            if year is not None:
                df_imp = df[cols][str(year)]
            else:
                df_imp = df[cols]
            
            df_imp = df_imp.div(df_imp.sum(axis=1), axis=0) # Divide by sum of row
            df_imp = df_imp.mean() # Take mean over time
            
            if only_big:
                for row in df_imp.index:
                    country = row.split('_')[0]
                    if country not in bigs:
                        df_imp = df_imp.drop(row)
            df_imp /= df_imp.sum() # Remove this if big shouldn't be normalized to 1
            
            df_imp.index = [col.split('_')[0] for col in df_imp.index]
            df_w[importer] = df_imp
            
            if only_big: df_w = df_w.fillna(0.0)

    # Column is importer and row (index) is where the import is from
    # Column is exporter and row (index) is where the export is to

    return df_w

def create_trade_var(df_w, years=False, cont_countries=False,
                     tradevar='Y', trend_filter='Trend', from_web=False):
    if from_web:
        get_oecd_data(large=True)

    # Create GDP dataframe
    var_mask = list(set([tradevar] + ['Y']))
    df = filter_for_panel(trend_filter=trend_filter, var_mask=var_mask,
                          large=True)

    # Drop columns which don't have much data
    for col in df:
        if col.split('_')[0] != tradevar:
            df = df.drop(col, axis=1)
    for col in df:
        nobs = 96
        if trend_filter == 'Hamilton': nobs -= 11
        if len(df[col].dropna()) < nobs: # 96
            df = df.drop(col, axis=1)
    
    countries = set([col.split('_')[1] for col in df])
    common_countries = countries.intersection(set(df_w.index))

    for col in df:
        if col.split('_')[1] not in common_countries:
            df = df.drop(col, axis=1)
            
    if years:
        df_ws = {}
        for year in range(1995,2019):
            df_w = create_weights(year=year)
            
            df_w = df_w[common_countries]
            df_w = df_w[df_w.index.isin(common_countries)]
            df_w = df_w/df_w.sum()
            df_w = df_w.reindex(sorted(df_w.columns), axis=1)
            df_w = df_w.reindex(sorted(df_w.index))
            
            df_ws.update({year: df_w})

    if years:
        # Only common countries should be included in the weight matrix
        df_w = df_w[common_countries]
        df_w = df_w[df_w.index.isin(common_countries)]
        df_w = df_w/df_w.sum()
    
        # Re-index the dataframes so they match
        df = df.reindex(sorted(df.columns), axis=1)
        df_w = df_w.reindex(sorted(df_w.columns), axis=1)
        df_w = df_w.reindex(sorted(df_w.index))
    else:
        df = df.reindex(sorted(df.columns), axis=1)

    df_Ys = pd.DataFrame(index=df.index, columns=df_w.columns)

    if cont_countries:
        T = len(df)
        for country in common_countries:
            for t in range(T):
                if years:
                    w_year = min(max(df_Ys.index[t].year, 1995), 2018)
                    df_w = df_ws[w_year]
                
                cols = [col for col in df if not np.isnan(df[col].iloc[t])]
                cols_w = [col.split('_')[1] for col in cols]
                df_country_w = df_w[country][df_w.index.isin(cols_w)]
                df_country_w = df_country_w/df_country_w.sum()
                
                df_Ys[country].iloc[t] = df[cols].iloc[t] @ np.array(df_country_w)
                
        for col in df_Ys:
            df_Ys[col] = pd.to_numeric(df_Ys[col])
    else:
        for country in common_countries:
            col_length = len(df[f'{tradevar}_{country}'].dropna())
            cols = []
            while len(cols) < 5:
                cols = [col for col in df if len(df[col].dropna()) >= col_length]
                col_length -= 1
            cols_w = [col.split('_')[1] for col in cols]
            df_country_w = df_w[country][df_w.index.isin(cols_w)]
            df_country_w = df_country_w/df_country_w.sum()
            df_Ys[country] = df[cols] @ np.array(df_country_w)
            
    for col in df_Ys:
        df_Ys[col] = pd.to_numeric(df_Ys[col])
            
    return df_Ys

def do_trade_var(varname, tradevar_w, lags, N_countries, ordered_countries, T, df, df_Ys):
    df[varname] = np.nan
    for lag in range(lags):
        df[f'{tradevar_w}_{lag+1}_star'] = np.nan
        
    for i in range(N_countries):
        country = ordered_countries[i]
        if country in df_Ys:
            df[varname].iloc[i*T:(i+1)*T] = df_Ys[country]
            for lag in range(lags):
                var_l = df_Ys[country].shift(lag+1)
                df[f'{tradevar_w}_{lag+1}_star'].iloc[i*T:(i+1)*T] = var_l
                
    return df

def add_trade_var(df, df_w, lags, years=False, tradevar='Y',
                  cont_countries=False, trend_filter='Trend', h=None):
    df_Ys = create_trade_var(df_w, tradevar=tradevar,
                             cont_countries=cont_countries, years=years,
                             trend_filter=trend_filter)
    
    ordered_countries = []
    for country in df['country']:
        if country not in ordered_countries:
            ordered_countries.append(country)
    N_countries = len(set(ordered_countries))
    T = len(df)//N_countries
    
    df = do_trade_var(f'{tradevar}_star', tradevar, lags, N_countries, ordered_countries, T, df, df_Ys)
                
    if h is not None:
        for col in df_Ys:
            df_Ys[col] = df_Ys[col].rolling(h+1).sum().shift(-h)
            
        df = do_trade_var(f'{tradevar}cum_star', f'{tradevar}cum', lags, N_countries, ordered_countries, T, df, df_Ys)
                         
    return df

def save_for_matlab(pp=4, trend_filter='Trend'):
    for var in ['Y','YT','YNT','C','CT','CNT','E','RR','P','EX','IM','NX']:
        var_mask = [var,'P','R','RR']
        if var == 'NX':
            var_mask += ['EX','IM']
        df = create_reg_data(1, pp, trend_filter, var_mask, Ys=['Y','P','RR'], drop=True,
                             quiet=True)
        
        Y = df[[var,'Y_star','P_star','RR_star']]
        X_list = []
        for i in range(pp):
            X_list += [f'{var}_{i+1}',f'Y_{i+1}_star',f'P_{i+1}_star',f'RR_{i+1}_star']
        X = df[X_list]
        X['cons'] = 1.0
        Y.to_csv(f'Data\MATLAB\Y_{var}.csv')
        X.to_csv(f'Data\MATLAB\X_{var}.csv', index=False)
        
        countries = list(set(df['country']))
        with open(f'Data\MATLAB\countries_{var}.txt', 'w') as f:
            for i,country in enumerate(countries):
                if i == 0:
                    f.write(country)
                else:
                    f.write(f'\n{country}')
        
        for country in countries:
            dfc = df[df['country']==country]
            Y = dfc[[var,'Y_star','P_star','RR_star']]
            X = dfc[X_list]
            X['cons'] = 1.0
            
            Y.to_csv(f'Data\MATLAB\Y_{country}_{var}.csv')
            X.to_csv(f'Data\MATLAB\X_{country}_{var}.csv', index=False)
    
def add_country_Ys_dummies(df, max_h, pp, tradevar='Y'):
    # Add dummy interaction
    df_country_D = pd.get_dummies(df['country'], drop_first=False)
    countries = list(set(df_country_D.columns))
    for country in countries:
        df[f'{tradevar}_star_{country}'] = df[f'{tradevar}_star']*df_country_D[country]
        for i in range(max_h+pp-1):
            df[f'{tradevar}_{i+1}_star_{country}'] = df[f'{tradevar}_{i+1}_star']*df_country_D[country]
    return df