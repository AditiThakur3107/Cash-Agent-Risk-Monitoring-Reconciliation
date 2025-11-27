"""
Simple rule engine with in-memory rule store.
Rules are simple functions that set a 'predicted_label' on the dataframe.
We persist histories in a JSON file 
"""
import json
from typing import List, Dict
import pandas as pd

DEFAULT_RULES = {
    "high_amount_debit": {
        "desc": "Debit transactions > 10k flagged as suspicious_pattern_1",
        "threshold": 10000.0,
        "action": "suspicious_pattern_1"
    },
    "frequent_small_credits": {
        "desc": "Many small credits in short interval -> suspicious_pattern_2",
        "count": 5,
        "window_minutes": 60,
        "action": "suspicious_pattern_2"
    }
}

class RuleStore:
    def __init__(self, rules=None):
        self.rules = rules or DEFAULT_RULES.copy()
        self.history = []

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['predicted_label'] = 'normal'
        # rule 1: high amount debit
        r = self.rules.get("high_amount_debit")
        mask = (df['type']=='debit') & (df['amount'] >= r['threshold'])
        df.loc[mask, 'predicted_label'] = r['action']
        # rule 2: frequent small credits - naive implementation
        r2 = self.rules.get("frequent_small_credits")
        df_sorted = df.sort_values(['account_id','timestamp'])
        # rolling count of small credits per account
        small_credit_mask = (df_sorted['type']=='credit') & (df_sorted['amount'] < 500.0)
        # simplistic: mark accounts with many small credits as suspicious for all their txns
        counts = df_sorted.loc[small_credit_mask].groupby('account_id').size()
        flagged_accounts = counts[counts >= r2['count']].index.tolist()
        df.loc[df['account_id'].isin(flagged_accounts), 'predicted_label'] = r2['action']
        return df

    def update_rule(self, rule_name, param_name, value):
        if rule_name in self.rules:
            self.rules[rule_name][param_name] = value
            self.history.append({"rule":rule_name, "param":param_name, "value":value})
        else:
            raise KeyError("Rule not found")

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump({"rules": self.rules, "history": self.history}, f, indent=2)
