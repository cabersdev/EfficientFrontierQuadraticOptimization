import pandas as pd
import numpy as np
from data_fetcher import fetch_data, params, validate_data

def fetch_all_data() -> pd.DataFrame:
    """Fetcha tutti i ticker e crea un unico DataFrame"""
    all_data = []
    
    for ticker in params.tickers:
        data = fetch_data(ticker)
        if data is not None and validate_data(data):
            all_data.append(data)
    
    if not all_data:
        raise ValueError("Nessun dato valido è stato fetchato")
    
    return pd.concat(all_data, ignore_index=True)

df = fetch_all_data()
    
def get_clean_dataframe(df):
    
    df = df.sort_values(['Ticker', 'Date'])
    df = df.drop_duplicates(['Ticker', 'Date'])
    
    colonne = ['Open', 'High', 'Low', 'Close', 'Volume']
    df[colonne] = df[colonne].apply(pd.to_numeric, errors='coerce')
    df['Date'] = pd.to_datetime(df['Date'])
    
    df = df.dropna(['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker'])
    
    return df

clean_df = get_clean_dataframe()