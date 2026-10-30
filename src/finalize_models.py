"""
Retrains the VALIDATION-SELECTED models on all labelled data and saves them for
inference. The model types are read from reports/model_selection.json - they are
not hardcoded here, so the saved models are always the ones chosen on validation.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd

from common import (CAT_BASELINES, CAT_FEATURES, EARLY_WINDOW_DAYS, RANDOM_SEED, REMOVED_FEATURES,
                    RISK_LABEL, STAGE1_NUM_FEATURES, STAGE2_BASE_NUM_FEATURES,
                    STAGE2_NUM_FEATURES_WITH_EFFORT, add_full_text, build_effort_pipeline,
                    build_risk_pipeline, load_selection, oof_effort_predictions, risk_population,
                    to_dense)

N_SHAP_BACKGROUND = 200


def main():
    print("=" * 80)
    print("FINALIZING PRODUCTION MODELS (validation-selected, trained on ALL labelled data)")
    print("=" * 80)

    selection = load_selection()
    effort_name, risk_family = selection["effort_model"], selection["risk_family"]
    print(f"Stage 1 model (selected on validation): {effort_name}")
    print(f"Stage 2 model (selected on validation): {selection['risk_model']}")

    df = add_full_text(pd.read_csv("data/processed/xd_modeling_dataset.csv"))
    print(f"\nTotal issues available: {len(df)}")
    print(f"Split breakdown being combined: {df['split'].value_counts().to_dict()}")

    # -------------------------------------------------------------------------
    # STAGE 1: effort model, log1p target, every issue with a Story Points value
    # -------------------------------------------------------------------------
    print(f"\n--- Stage 1: Effort model ({effort_name}) ---")
    all_effort_df = df.dropna(subset=["story_points"]).copy().reset_index(drop=True)
    stage1_pipe = build_effort_pipeline(effort_name)
    stage1_pipe.fit(all_effort_df, np.log1p(all_effort_df["story_points"].values))

    os.makedirs("models", exist_ok=True)
    joblib.dump(stage1_pipe, "models/stage1_effort_model.joblib")
    print(f"Trained on {len(all_effort_df)} issues. Saved: models/stage1_effort_model.joblib")

    # -------------------------------------------------------------------------
    # STAGE 2: risk model (+ predicted effort). Training rows get out-of-fold
    # predicted effort so no row ever sees a prediction from a model trained on it.
    # -------------------------------------------------------------------------
    print(f"\n--- Stage 2: Risk model ({selection['risk_model']}) ---")
    all_effort_df["pred_effort"] = oof_effort_predictions(all_effort_df, effort_name)
    risk_df = risk_population(all_effort_df)
    print(f"Risk-eligible issues (still open past day {EARLY_WINDOW_DAYS}): {len(risk_df)} "
          f"(At-risk: {risk_df[RISK_LABEL].mean():.1%})")

    stage2_pipe = build_risk_pipeline(risk_family, with_effort=True)
    stage2_pipe.fit(risk_df, risk_df[RISK_LABEL].values)
    joblib.dump(stage2_pipe, "models/stage2_risk_model.joblib")
    print(f"Trained on {len(risk_df)} eligible issues. Saved: models/stage2_risk_model.joblib")

    # Background sample for SHAP at inference time (needed by LinearExplainer;
    # harmless for tree models). Stored already transformed.
    background = risk_df.sample(n=min(N_SHAP_BACKGROUND, len(risk_df)), random_state=RANDOM_SEED)
    joblib.dump(to_dense(stage2_pipe.named_steps["preprocessor"].transform(background)),
                "models/stage2_shap_background.joblib")
    print("Saved: models/stage2_shap_background.joblib")

    # -------------------------------------------------------------------------
    # Config + example tickets for the predict.py sanity check
    # -------------------------------------------------------------------------
    config = {
        "early_window_days": EARLY_WINDOW_DAYS,
        "cat_features": CAT_FEATURES,
        "cat_baselines": CAT_BASELINES,
        "stage1_num_features": STAGE1_NUM_FEATURES,
        "stage2_base_num_features": STAGE2_BASE_NUM_FEATURES,
        "stage2_num_features_with_effort": STAGE2_NUM_FEATURES_WITH_EFFORT,
        "removed_features": REMOVED_FEATURES,
        "stage1_model": effort_name,
        "stage2_model": selection["risk_model"],
        "selection_source": "reports/model_selection.json (validation split)",
        "n_effort_training_rows": int(len(all_effort_df)),
        "n_risk_training_rows": int(len(risk_df)),
        "risk_at_risk_rate": float(risk_df[RISK_LABEL].mean()),
    }
    with open("models/model_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("\nSaved: models/model_config.json")

    sample_cols = (["key", "story_points", RISK_LABEL] + CAT_FEATURES + STAGE2_BASE_NUM_FEATURES
                   + ["num_components", "num_labels", "summary", "description"])
    sample_cols = list(dict.fromkeys(c for c in sample_cols if c in risk_df.columns))
    risk_df.sample(n=min(5, len(risk_df)), random_state=RANDOM_SEED)[sample_cols].to_csv(
        "models/example_tickets_for_sanity_check.csv", index=False)
    print("Saved: models/example_tickets_for_sanity_check.csv")

    print("\n" + "=" * 80)
    print("Production models finalized. Both saved pipelines include their own")
    print("preprocessing (encoding, scaling, TF-IDF): raw ticket fields in, prediction out.")
    print("=" * 80)


if __name__ == "__main__":
    main()
