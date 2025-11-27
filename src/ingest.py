import pandas as pd

def load_transactions(path):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    # basic type normalization
    df['amount'] = df['amount'].astype(float)
    return df
