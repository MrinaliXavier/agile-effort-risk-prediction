import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.base import clone

from lightgbm import LGBMRegressor, LGBMClassifier

EARLY_WINDOW_DAYS = 5

CAT_FEATURES = ["issuetype", "priority"]

STAGE1_NUM_FEATURES = [
    "num_components", "num_labels", "summary_len", "description_len",
    "reporter_prev_issue_count", "priority_ordinal",
    "has_description", "has_components", "has_labels",
]

STAGE2_BASE_NUM_FEATURES = STAGE1_NUM_FEATURES + [
    "early_status_changes", "early_assignee_changes", "early_priority_changes",
    "early_comment_count", "had_early_status_change", "first_status_change_days_imputed",
]

STAGE2_NUM_FEATURES_WITH_EFFORT = STAGE2_BASE_NUM_FEATURES + ["pred_effort"]


def build_stage1_preprocessor():
    return ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
        ("num", StandardScaler(), STAGE1_NUM_FEATURES),
        ("text", TfidfVectorizer(max_features=300, stop_words="english", ngram_range=(1, 2), sublinear_tf=True), "full_text"),
    ])


def build_stage2_preprocessor():
    return ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
        ("num", StandardScaler(), STAGE2_NUM_FEATURES_WITH_EFFORT),
    ])


def main():
    print("=" * 80)
    print("FINALIZING PRODUCTION MODELS (trained on ALL labelled data)")
    print("=" * 80)

    df = pd.read_csv("data/processed/xd_modeling_dataset.csv")
    df["full_text"] = df["summary"].fillna("") + " " + df["description"].fillna("")

    print(f"Total issues available: {len(df)}")
    print(f"Split breakdown being combined: {df['split'].value_counts().to_dict()}")

    # -------------------------------------------------------------------------
    # STAGE 1: FINAL EFFORT MODEL - LightGBM (Tabular + TF-IDF), log1p target
    # trained on every issue that has a Story Points value, across all splits
    # -------------------------------------------------------------------------
    print("\n--- Stage 1: Effort model (LightGBM, Tabular + TF-IDF) ---")

    all_effort_df = df.dropna(subset=["story_points"]).copy().reset_index(drop=True)
    y_effort_log = np.log1p(all_effort_df["story_points"].values)

    stage1_pipe = Pipeline([
        ("preprocessor", build_stage1_preprocessor()),
        ("model", LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, verbose=-1)),
    ])
    stage1_pipe.fit(all_effort_df, y_effort_log)

    os.makedirs("models", exist_ok=True)
    joblib.dump(stage1_pipe, "models/stage1_effort_model.joblib")
    print(f"Trained on {len(all_effort_df)} issues. Saved: models/stage1_effort_model.joblib")

    # -------------------------------------------------------------------------
    # STAGE 2: FINAL RISK MODEL - LightGBM (+ Pred Effort)
    # Needs leak-free pred_effort for every eligible row used in training, so we
    # generate it via 5-fold out-of-fold prediction across ALL effort-labelled
    # rows (same technique used throughout - no row ever sees its own prediction).
    # -------------------------------------------------------------------------
    print("\n--- Stage 2: Risk model (LightGBM, + Pred Effort) ---")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_pred_effort = np.zeros(len(all_effort_df))
    for tr_idx, te_idx in kf.split(all_effort_df):
        tr_f = all_effort_df.iloc[tr_idx]
        te_f = all_effort_df.iloc[te_idx]
        fold_pipe = Pipeline([
            ("preprocessor", build_stage1_preprocessor()),
            ("model", LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, verbose=-1)),
        ])
        fold_pipe.fit(tr_f, np.log1p(tr_f["story_points"].values))
        oof_pred_effort[te_idx] = np.expm1(fold_pipe.predict(te_f)).clip(min=0)
    all_effort_df["pred_effort"] = oof_pred_effort

    risk_df = all_effort_df[
        (all_effort_df["risk_25pct"].notna()) & (all_effort_df["eligible_for_risk_model"] == 1)
    ].copy().reset_index(drop=True)
    risk_df["risk_25pct"] = risk_df["risk_25pct"].astype(int)

    print(f"Risk-eligible issues (still open past day {EARLY_WINDOW_DAYS}): {len(risk_df)} "
          f"(At-risk: {risk_df['risk_25pct'].mean():.1%})")

    stage2_pipe = Pipeline([
        ("preprocessor", build_stage2_preprocessor()),
        ("model", LGBMClassifier(n_estimators=100, is_unbalance=True, random_state=42, verbose=-1)),
    ])
    stage2_pipe.fit(risk_df, risk_df["risk_25pct"].values)

    joblib.dump(stage2_pipe, "models/stage2_risk_model.joblib")
    print(f"Trained on {len(risk_df)} eligible issues. Saved: models/stage2_risk_model.joblib")

    # -------------------------------------------------------------------------
    # Save feature config + a few known tickets for the later sanity check
    # -------------------------------------------------------------------------
    config = {
        "early_window_days": EARLY_WINDOW_DAYS,
        "cat_features": CAT_FEATURES,
        "stage1_num_features": STAGE1_NUM_FEATURES,
        "stage2_base_num_features": STAGE2_BASE_NUM_FEATURES,
        "stage2_num_features_with_effort": STAGE2_NUM_FEATURES_WITH_EFFORT,
        "stage1_model": "LightGBM (Tabular + TF-IDF)",
        "stage2_model": "LightGBM (+ Pred Effort)",
        "n_effort_training_rows": int(len(all_effort_df)),
        "n_risk_training_rows": int(len(risk_df)),
        "risk_at_risk_rate": float(risk_df["risk_25pct"].mean()),
    }
    with open("models/model_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("\nSaved: models/model_config.json")

    # Save a handful of real example tickets (with their known outcomes) so the
    # next step (inference sanity check) can confirm predict.py reproduces
    # sensible results on data the models have actually seen.
    sample_cols = ["key", "story_points", "risk_25pct"] + CAT_FEATURES + STAGE2_BASE_NUM_FEATURES + ["summary", "description"]
    sample_cols = [c for c in sample_cols if c in risk_df.columns]
    examples = risk_df.sample(n=min(5, len(risk_df)), random_state=42)[sample_cols]
    examples.to_csv("models/example_tickets_for_sanity_check.csv", index=False)
    print("Saved: models/example_tickets_for_sanity_check.csv")

    print("\n" + "=" * 80)
    print("Production models finalized. Both saved pipelines include their own")
    print("preprocessing (encoding, scaling, TF-IDF) - no separate preprocessing")
    print("step is needed at inference time, just raw ticket fields in, prediction out.")
    print("=" * 80)


if __name__ == "__main__":
    main()