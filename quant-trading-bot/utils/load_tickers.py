# utils/load_tickers.py
import pandas as pd

def load_ticker(path="TICKER.csv"):
    df = pd.read_csv(path, header=None)
    tickers = df.values.flatten()
    tickers = pd.Series(tickers).dropna().astype(str)
    tickers = tickers.str.strip().str.upper().tolist()
    return tickers[0] if tickers else None



def load_tickers(path="TICKERS.csv"):
    df = pd.read_csv(path, header=None)
    tickers = df.values.flatten()
    tickers = pd.Series(tickers).dropna().astype(str)
    return tickers.str.strip().str.upper().tolist()
