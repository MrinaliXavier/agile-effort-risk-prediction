"""
mp_finalize_models.py - train and save the FINAL per-project models for prediction.

For each project the model choice is read, not made, here: Stage 1 effort model and Stage 2
family come from reports/mp_<label>_results.json (selected on VALIDATION by
mp_train_evaluate.py); Stage 2 uses the proposed variant '+ Team SP + Model SP'.
Both stages are then retrained on ALL labelled tickets of the project (Stage 2 training rows
get out-of-fold Stage 1 estimates, as everywhere else).

Saved to models/<label>/<PROJECT>/:
  stage1.joblib, stage2.joblib     full pipelines (raw ticket fields in, prediction out)
  background1.joblib, background2.joblib   SHAP reference samples (already transformed)
  config.json                      choices, features, reference categories, spill rate
  example_tickets.csv              5 tickets for a smoke test of mp_predict.py (they were
                                   in the training data - not an accuracy check)

Run from the project root, after mp_train_evaluate.py:
    python src/mp_finalize_models.py              # spillover label
    python src/mp_finalize_models.py --label residual
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

from common import (CAT_FEATURES, PROPOSED_VARIANT, RANDOM_SEED, add_estimate_features, add_full_text,
                    build_effort_pipeline, build_risk_pipeline, make_risk_variants, most_common_baselines,
                    oof_effort_predictions, to_dense)
from mp_train_evaluate import LABELS, population

N_BACKGROUND = 200
EXAMPLE_COLUMNS = ["key", "summary", "description", "issuetype", "priority", "num_components", "num_labels",
                   "reporter_prev_issue_count", "early_status_changes", "early_assignee_changes",
                   "early_priority_changes", "early_comment_count", "first_status_change_days_imputed",
                   "backlog_days", "sprint_length_days", "story_points"]


def finalize_project(df, key, sel, cfg, out_dir):
    label, team_col = cfg["label"], cfg["team_col"]
    features = make_risk_variants(cfg["base"])[PROPOSED_VARIANT]
    df = add_full_text(df).reset_index(drop=True)
    baselines = most_common_baselines(df)

    stage1 = build_effort_pipeline(sel["effort_model"], baselines)
    stage1.fit(df, np.log1p(df["story_points"].values))
    df["pred_effort"] = oof_effort_predictions(df, sel["effort_model"], cat_baselines=baselines)
    done = population(add_estimate_features(df, team_col), label)
    stage2 = build_risk_pipeline(sel["risk_family"], cat_baselines=baselines, num_features=features)
    stage2.fit(done, done[label].values)

    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(stage1, f"{out_dir}/stage1.joblib")
    joblib.dump(stage2, f"{out_dir}/stage2.joblib")
    bg1 = df.sample(n=min(N_BACKGROUND, len(df)), random_state=RANDOM_SEED)
    bg2 = done.sample(n=min(N_BACKGROUND, len(done)), random_state=RANDOM_SEED)
    joblib.dump(to_dense(stage1.named_steps["preprocessor"].transform(bg1)), f"{out_dir}/background1.joblib")
    joblib.dump(to_dense(stage2.named_steps["preprocessor"].transform(bg2)), f"{out_dir}/background2.joblib")

    config = {"project": key, "label": label, "team_col": team_col,
              "effort_model": sel["effort_model"], "risk_family": sel["risk_family"],
              "risk_variant": PROPOSED_VARIANT, "stage2_features": features,
              "reference_categories": dict(zip(CAT_FEATURES, baselines)),
              "selection_source": f"reports/mp_{cfg['label_name']}_results.json (validation split)",
              "n_effort_rows": int(len(df)), "n_risk_rows": int(len(done)),
              "historical_delay_rate": round(float(done[label].mean()), 4)}
    with open(f"{out_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)
    examples = df.sample(n=min(5, len(df)), random_state=RANDOM_SEED)
    cols = [c for c in EXAMPLE_COLUMNS + [team_col, label] if c in examples.columns]
    examples[cols].to_csv(f"{out_dir}/example_tickets.csv", index=False)
    return config


def main():
    args = sys.argv[1:]
    label_name = "spillover"
    if "--label" in args:
        i = args.index("--label")
        label_name, args = args[i + 1], args[:i] + args[i + 2:]
    cfg = dict(LABELS[label_name], label_name=label_name)
    with open(f"reports/mp_{label_name}_results.json") as f:
        selection = {r["project"]: r for r in json.load(f)["per_project"] if "risk_family" in r}
    data = pd.read_csv(cfg["dataset"])
    data[CAT_FEATURES] = data[CAT_FEATURES].fillna("None").astype(str)
    only = {k.upper() for k in args}

    for key, sel in selection.items():
        if only and key not in only:
            continue
        out_dir = f"models/{label_name}/{key}"
        try:
            c = finalize_project(data[data["project"] == key].copy(), key, sel, cfg, out_dir)
        except Exception as exc:
            print(f"{key}: !! failed: {exc}")
            continue
        print(f"{key}: Stage 1 {c['effort_model']} | Stage 2 {c['risk_family']} ({c['risk_variant']}) | "
              f"{c['n_risk_rows']} tickets | historical delay rate {c['historical_delay_rate']:.0%} -> {out_dir}/")


if __name__ == "__main__":
    main()
