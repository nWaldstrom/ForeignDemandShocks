import numpy as np
import pandas as pd
import statsmodels.api as sm

def filter_data(df, method, drop_n_start=0, drop_n_end=0, **kwargs):
    assert method in ['HP', 'BK', 'CF', 'Trend', 'Hamilton']
    assert isinstance(drop_n_start, int)
    assert isinstance(drop_n_end, int)
    
    def fill_kwargs(def_kwargs, kwargs):
        for arg in def_kwargs:
            if arg not in kwargs:
                kwargs[arg] = def_kwargs[arg]
        return kwargs
                
    df_dropped = df.dropna()

    # HP filter
    if method == 'HP':
        kwargs = fill_kwargs({'lamb': 1600}, kwargs)
        
        cycle, trend = sm.tsa.filters.hpfilter(df_dropped, kwargs['lamb'])

    # BK filter
    elif method == 'BK':
        kwargs = fill_kwargs({'low': 6, 'high':32, 'K': 12}, kwargs)
        
        cycle = sm.tsa.filters.bkfilter(df_dropped, kwargs['low'],
                                        kwargs['high'], kwargs['K'])
        trend = pd.DataFrame([np.nan]*len(cycle), index=cycle.index)

    # CF filter (trend is only really a trend if drift = False)
    elif method == 'CF':
        kwargs = fill_kwargs({'low': 6, 'high':32, 'drift': True}, kwargs)
        
        cycle, trend = sm.tsa.filters.cffilter(df_dropped, kwargs['low'],
                                               kwargs['high'], kwargs['drift'])

    # Exponential trend
    elif method == 'Trend':
        kwargs = fill_kwargs({'order': 4}, kwargs)
        
        Y = np.array(df_dropped)
        T = len(Y)
        X = [(np.arange(T, dtype=np.int64)**power).reshape(T,1) for power in
             range(kwargs['order']+1)]
        X = np.concatenate(X, axis=1)
        B = np.linalg.inv(X.T@X)@X.T@Y
        trend = pd.DataFrame(X@B, index=df_dropped.index)
        cycle = pd.DataFrame(Y.reshape(len(X),1)-np.array(trend),
                             index=df_dropped.index)
    
    # Hamilton regression
    elif method == 'Hamilton':
        kwargs = fill_kwargs({'h': 8, 'p': 4}, kwargs)
        
        X = np.concatenate([pd.DataFrame(df_dropped.shift(kwargs['h']+i)) for
                            i in range(kwargs['p'])], axis=1)
        start_idx = pd.DataFrame(X).first_valid_index() + kwargs['p']-1
        X = X[start_idx:]
        Y = np.array(df_dropped)[start_idx:]
        B = np.linalg.inv(X.T@X)@X.T@Y
        trend = pd.DataFrame(X@B, index=df_dropped.index[start_idx:])
        cycle = pd.DataFrame(Y.reshape(len(X),1)-np.array(trend),
                             index=df_dropped.index[start_idx:])
    
    # Drop variables if needed
    if drop_n_end > 0:
        cycle = pd.DataFrame(cycle[drop_n_start:-drop_n_end], index=df.index)
        trend = pd.DataFrame(trend[drop_n_start:-drop_n_end], index=df.index)
    else:
        cycle = pd.DataFrame(cycle[drop_n_start:], index=df.index)
        trend = pd.DataFrame(trend[drop_n_start:], index=df.index)
        
    return cycle, trend