"""
Simulate stakeholder feedback and drive rule updates.
We randomise feedback correctness; if many confirmations are received for a predicted label,
we keep the rule; if many rejections, we adjust the rule parameter.
"""
import random
import pandas as pd
from collections import Counter

def simulate_feedback(df_pred: pd.DataFrame, confirm_rate=0.8):
    """
    For each suspicious transaction, simulate a stakeholder response: confirm or reject.
    Returns: DataFrame with columns ['transaction_id','predicted_label','feedback'].
    """
    suspicious = df_pred[df_pred['predicted_label']!='normal']
    rows = []
    for _, r in suspicious.iterrows():
        confirmed = random.random() < confirm_rate
        feedback = 'confirm' if confirmed else 'reject'
        rows.append({"transaction_id": r['transaction_id'], "predicted_label": r['predicted_label'], "feedback": feedback})
    return pd.DataFrame(rows)

def process_feedback_and_update_rules(rule_store, feedback_df, min_reject_ratio=0.3):
    # Compute per-label feedback summary
    summary = feedback_df.groupby('predicted_label')['feedback'].value_counts().unstack(fill_value=0)
    for label in summary.index:
        total = summary.loc[label].sum()
        rejects = summary.loc[label].get('reject', 0)
        reject_ratio = rejects/total if total>0 else 0
        # If too many rejects, tweak rule threshold (simple heuristic)
        if reject_ratio > min_reject_ratio:
            # Example: make high_amount_debit more strict
            if label == 'suspicious_pattern_1':
                # raise threshold by 10%
                curr = rule_store.rules.get('high_amount_debit', {})
                new_threshold = curr.get('threshold',10000)*1.1
                rule_store.update_rule('high_amount_debit','threshold', new_threshold)
    return rule_store
