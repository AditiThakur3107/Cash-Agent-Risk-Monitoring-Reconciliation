import os
import argparse
from src.ingest import load_transactions
from src.rules import RuleStore
from src.ml_model import augment_with_ml
from src.feedback_simulator import simulate_feedback, process_feedback_and_update_rules
from src.reconciliation import reconcile_account, fallback_scan
import pandas as pd
import json

def run_pipeline(input_csv, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    print("Loading transactions...")
    df = load_transactions(input_csv)
    rule_store = RuleStore()
    print("Applying rules...")
    df_pred = rule_store.apply(df)
    print("(Optional) augmenting with ML score...")
    df_pred = augment_with_ml(df_pred)
    # Save predicted
    pred_path = os.path.join(out_dir, "classified_transactions.csv")
    df_pred.to_csv(pred_path, index=False)
    print(f"Saved predictions to {pred_path}")
    # Simulate stakeholder feedback
    feedback = simulate_feedback(df_pred, confirm_rate=0.85)
    feedback_path = os.path.join(out_dir, "simulated_feedback.csv")
    feedback.to_csv(feedback_path, index=False)
    # Process feedback & possibly update rules
    rule_store = process_feedback_and_update_rules(rule_store, feedback)
    # persist rules
    rule_store.to_json(os.path.join(out_dir, "rule_history.json"))
    # Reconciliation per account (naive)
    recon_reports = []
    unmatched_global = []
    for acc, g in df_pred.groupby('account_id'):
        _, unmatched_debits, unmatched_credits, report = reconcile_account(g)
        report['account_id'] = acc
        recon_reports.append(report)
        if len(unmatched_debits)>0:
            unmatched_global.append(unmatched_debits)
    # fallback scan
    if len(unmatched_global):
        unmatched_df = pd.concat(unmatched_global, ignore_index=True)
        suggestions = fallback_scan(df_pred, unmatched_df)
        with open(os.path.join(out_dir, "reconciliation_suggestions.json"), "w") as f:
            json.dump(suggestions, f, indent=2, default=str)
    # Save reconciliation summary
    recon_df = pd.DataFrame(recon_reports)
    recon_df.to_csv(os.path.join(out_dir, "reconciliation_report.csv"), index=False)
    print("Pipeline complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    run_pipeline(args.input, args.out)
