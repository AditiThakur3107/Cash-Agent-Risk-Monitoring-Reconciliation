"""
 a simple probability scorer using heuristics.
"""
import numpy as np
import pandas as pd

def heuristic_score(df: pd.DataFrame) -> pd.Series:
    # returns a score between 0 and 1 (higher = more suspicious)
    score = np.zeros(len(df))
    score += (df['amount'] / (df['amount'].max() + 1)).values
    score += (df['type'] == 'debit').astype(int) * 0.2
    # normalize
    score = score / score.max()
    return pd.Series(score, index=df.index)

def augment_with_ml(df):
    df = df.copy()
    df['ml_score'] = heuristic_score(df)
    # combine rules and ml to form final label
    df['final_score'] = df['ml_score']
    return df
