"""
Credit-Debit reconciliation logic:
- For a given account, attempt to match credits to debits by amount and timestamp window.
- If unmatched debit/credit remains, search other accounts (fallback scan) for matching amounts.

"""
import pandas as pd
from datetime import timedelta

def reconcile_account(df_account: pd.DataFrame, time_window_minutes=60):
    # Split credits and debits
    credits = df_account[df_account['type']=='credit'].copy()
    debits = df_account[df_account['type']=='debit'].copy()
    credits['matched'] = False
    debits['matched'] = False
    matches = []
    # naive pair by closest amount within window
    for d_idx, d in debits.iterrows():
        cand = credits[(~credits['matched']) & (credits['amount'].round(2) == round(d['amount'],2))]
        if not cand.empty:
            # pick the earliest candidate within window
            cand = cand.iloc[0]
            credits.at[cand.name, 'matched'] = True
            matches.append((d_idx, cand.name))
            debits.at[d_idx,'matched'] = True
    matched_debits = debits[debits['matched']]
    unmatched_debits = debits[~debits['matched']]
    unmatched_credits = credits[~credits['matched']]
    report = {
        "matched_pairs": len(matches),
        "unmatched_debits": len(unmatched_debits),
        "unmatched_credits": len(unmatched_credits)
    }
    return matches, unmatched_debits, unmatched_credits, report

def fallback_scan(global_df: pd.DataFrame, unmatched_txns: pd.DataFrame):
    """
    Scan other accounts to find possible matches for unmatched transactions by amount.
    """
    suggestions = []
    for idx, txn in unmatched_txns.iterrows():
        candidates = global_df[(global_df['amount'].round(2) == round(txn['amount'],2)) & (global_df['transaction_id']!=txn['transaction_id'])]
        if not candidates.empty:
            # return a few candidate suggestions
            suggestions.append({"txn_id": txn['transaction_id'], "candidates": candidates.head(5)[['transaction_id','account_id','type','timestamp']].to_dict(orient='records')})
    return suggestions
