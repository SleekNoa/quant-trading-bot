# utils/load_tickers.py
import pandas as pd
import os

def _get_project_root():
    """Get the project root directory (parent of utils folder)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _find_csv(filename):
    """Find CSV file in project root or fallback to current directory."""
    project_root = _get_project_root()
    candidate = os.path.join(project_root, filename)
    if os.path.exists(candidate):
        return candidate
    return filename  # fallback to current directory

def load_ticker(path=None):
    if path is None:
        path = _find_csv("TICKER.csv")
    df = pd.read_csv(path, header=None)
    tickers = df.values.flatten()
    tickers = pd.Series(tickers).dropna().astype(str)
    tickers = tickers.str.strip().str.upper().str.strip('"').tolist()
    return tickers[0] if tickers else None



def load_tickers(path=None):
    if path is None:
        path = _find_csv("TICKERS.csv")
    df = pd.read_csv(path, header=None)
    tickers = df.values.flatten()
    tickers = pd.Series(tickers).dropna().astype(str)
    tickers = tickers.str.strip().str.upper().str.strip('"').tolist()
    return tickers
