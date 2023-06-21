import requests
import pandas as pd

def get_url(dataset, dimensions, starttime, endtime):
    url = f'http://stats.oecd.org/sdmx-json/data/{dataset}/'
    
    for dimension in dimensions:
        for item in dimension:
            url += f'{item}+'
        url = url[:-1] # Remove extra +
        url += '.'
    url = url[:-1] # Remove extra .

    url += f'/all?startTime={starttime}&endTime={endtime}'
    
    return url

def get_data_from_url(url):
    response = requests.get(url)
    data = response.json()
    
    return data

def get_data(dataset, dimensions, starttime, endtime):
    url = get_url(dataset, dimensions, starttime, endtime)
    data = get_data_from_url(url)
    
    return data

def get_df(dataset, dimensions, vardict, banned_countries, starttime, endtime):
    if dataset == 'TIVA_2021_C1':
        id1 = 1
        id2 = 2
        def slicer(i,j): return f'0:{i}:{j}:0'
    else:
        id1 = 0
        id2 = 1
        def slicer(i,j): return f'{i}:{j}:0:0'
    
    data = get_data(dataset, dimensions, starttime, endtime)
    
    values_countries = data['structure']['dimensions']['series'][id1]['values']
    countries = [value['id'] for value in values_countries]

    values_subjects = data['structure']['dimensions']['series'][id2]['values']
    variables = [value['id'] for value in values_subjects]
    
    values_freqs = data['structure']['dimensions']['observation'][0]['values']
    freqs = [value['id'] for value in values_freqs]
    cols = []
    for i, country in enumerate(countries):
        for j, variable in enumerate(variables):
            if country not in banned_countries:
                cols.append(f'{variable}_{country}')
    df = pd.DataFrame(index=range(len(freqs)), columns=cols)

    series = data['dataSets'][0]['series']
    for i, country in enumerate(countries):
        for j, variable in enumerate(variables):
            if country not in banned_countries:
                try:
                    selected = series[slicer(i,j)]['observations']
                    ser = [item[1][0] for item in selected.items()]
                    idx = [int(key) for key in selected.keys()]
                    df[f'{variable}_{country}'] = pd.Series(ser, index=idx)
                except Exception:
                    continue
                    
    freq_map = {}
    for i, freq in enumerate(freqs):
        freq_map.update({i: freq})
    df.index = df.index.map(freq_map.get)
    df = df.sort_index()
    df.index = pd.to_datetime(df.index)
    
    if vardict is not None:
        for translation in vardict:
            df.columns = df.columns.str.replace(translation, vardict[translation])
        
    return df