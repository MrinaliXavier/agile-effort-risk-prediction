"""
PHASE 4 & 5: advanced models, two-stage ablation, SHAP - and MODEL SELECTION.

This is the ONLY script that chooses models. It chooses them on the VALIDATION
split and writes the choice to reports/model_selection.json. evaluate_test_set.py
and finalize_models.py read that file and never re-select, so the test set is
used only to report the performance of models that were already chosen.
"""
import json
import os

import numpy as np
import pandas as pd

from common import (EARLY_WINDOW_DAYS, EFFORT_CANDIDATES, RISK_CANDIDATES, RISK_LABEL,
                    RISK_VARIANTS, SELECTION_FILE, build_effort_pipeline, build_risk_pipeline,
                    add_full_text, eval_effort_metrics, eval_risk_metrics,
                    oof_effort_predictions, predict_effort, risk_name, risk_population,
                    save_selection, shap_values_positive_class)

EFFORT_SELECTION_METRIC = "MAE"      # lower is better
RISK_SELECTION_METRIC = "ROC-AUC"    # higher is better


def main():
    print("=" * 80)
    print("PHASE 4 & 5: ADVANCED MODELING (NLP, TWO-STAGE RISK & SHAP) + VALIDATION-ONLY SELECTION")
    print("=" * 80)

    df = add_full_text(pd.read_csv("data/processed/xd_modeling_dataset.csv"))
    train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
    val_df = df[df["split"] == "val"].copy().reset_index(drop=True)

    # -------------------------------------------------------------------------
    # PART 1: EFFORT ESTIMATION - tabular vs tabular + TF-IDF, log1p target
    # -------------------------------------------------------------------------
    print("\n" + "#" * 80)
    print("1. EFFORT PREDICTION: TABULAR vs TABULAR + TF-IDF (log1p target)")
    print("#" * 80)

    y_train_eff_raw = train_df["story_points"].values
    y_val_eff_raw = val_df["story_points"].values
    print(f"Story Points skew (raw): {pd.Series(y_train_eff_raw).skew():.2f} -> training on log1p")

    effort_results = {}
    for name in EFFORT_CANDIDATES:
        pipe = build_effort_pipeline(name)
        pipe.fit(train_df, np.log1p(y_train_eff_raw))
        effort_results[name] = {
            "train": eval_effort_metrics(y_train_eff_raw, predict_effort(pipe, train_df)),
            "val": eval_effort_metrics(y_val_eff_raw, predict_effort(pipe, val_df)),
        }

    effort_table = pd.DataFrame([
        {"Model": name, "Val MAE": r["val"]["MAE"], "Val RMSE": r["val"]["RMSE"],
         "Val MdAE": r["val"]["MdAE"], "Val R2": r["val"]["R2"],
         "Val PRED(25)%": r["val"]["PRED(25)%"], "Val MdMRE%": r["val"]["MdMRE%"],
         "Train R2": r["train"]["R2"]}
        for name, r in effort_results.items()
    ])
    print("\nEffort Regression Results (Validation Set, log1p-trained):")
    print(effort_table.to_string(index=False))

    best_effort_name = min(effort_results, key=lambda n: effort_results[n]["val"][EFFORT_SELECTION_METRIC])
    print(f"\n>> SELECTED Stage 1 model: '{best_effort_name}' (lowest Val {EFFORT_SELECTION_METRIC})")

    # -------------------------------------------------------------------------
    # PART 2: TWO-STAGE: EFFORT -> RISK (complete ablation on every model family)
    # -------------------------------------------------------------------------
    print("\n" + "#" * 80)
    print("2. TWO-STAGE MODELING: FEEDING PREDICTED EFFORT INTO RISK CLASSIFIER")
    print("#" * 80)

    # Train rows get out-of-fold predicted effort; validation rows get predictions
    # from the selected effort model fitted on the full training split.
    train_df["pred_effort"] = oof_effort_predictions(train_df, best_effort_name)
    full_effort_pipe = build_effort_pipeline(best_effort_name)
    full_effort_pipe.fit(train_df, np.log1p(train_df["story_points"].values))
    val_df["pred_effort"] = predict_effort(full_effort_pipe, val_df)

    train_done = risk_population(train_df)
    val_done = risk_population(val_df)
    print(f"Risk-eligible Train: {len(train_done)} (At-risk: {train_done[RISK_LABEL].mean():.1%})")
    print(f"Risk-eligible Val:   {len(val_done)} (At-risk: {val_done[RISK_LABEL].mean():.1%})")

    risk_results, fitted_risk_pipes = {}, {}
    for family in RISK_CANDIDATES:
        for variant in RISK_VARIANTS:
            name = risk_name(family, variant)
            pipe = build_risk_pipeline(family, with_effort=(variant == "+ Pred Effort"))
            pipe.fit(train_done, train_done[RISK_LABEL].values)
            fitted_risk_pipes[name] = pipe
            risk_results[name] = {
                "train": eval_risk_metrics(train_done[RISK_LABEL].values, pipe.predict(train_done),
                                           pipe.predict_proba(train_done)[:, 1]),
                "val": eval_risk_metrics(val_done[RISK_LABEL].values, pipe.predict(val_done),
                                         pipe.predict_proba(val_done)[:, 1]),
            }

    risk_table = pd.DataFrame([
        {"Model": name, "Val ROC-AUC": r["val"]["ROC-AUC"], "Val PR-AUC": r["val"]["PR-AUC"],
         "Val Macro-F1": r["val"]["Macro-F1"], "Val AtRisk-Recall": r["val"]["AtRisk-Recall"],
         "Val AtRisk-Precision": r["val"]["AtRisk-Precision"], "Val AtRisk-F1": r["val"]["AtRisk-F1"],
         "Val Brier": r["val"]["Brier"]}
        for name, r in risk_results.items()
    ])
    print("\nRisk Prediction Results - COMPLETE Ablation (With vs Without Predicted Effort):")
    print(risk_table.to_string(index=False))

    print("\n--- Ablation deltas on VALIDATION (ROC-AUC gained by adding predicted effort) ---")
    for family in RISK_CANDIDATES:
        no_eff = risk_results[risk_name(family, "No Effort")]["val"]["ROC-AUC"]
        with_eff = risk_results[risk_name(family, "+ Pred Effort")]["val"]["ROC-AUC"]
        print(f"  {family}: {no_eff:.4f} -> {with_eff:.4f}  (delta {with_eff - no_eff:+.4f})")

    # The system architecture is two-stage by design, so the Stage 2 ALGORITHM is
    # chosen among the "+ Pred Effort" variants. The "No Effort" variants remain
    # the ablation control for the selected algorithm.
    two_stage = {f: risk_results[risk_name(f, "+ Pred Effort")]["val"][RISK_SELECTION_METRIC] for f in RISK_CANDIDATES}
    best_risk_family = max(two_stage, key=two_stage.get)
    best_risk_name = risk_name(best_risk_family, "+ Pred Effort")
    best_overall = max(risk_results, key=lambda n: risk_results[n]["val"][RISK_SELECTION_METRIC])
    print(f"\n>> SELECTED Stage 2 model: '{best_risk_name}' (highest Val {RISK_SELECTION_METRIC} among two-stage variants)")
    if best_overall != best_risk_name:
        print(f"   Note: best variant overall on validation was '{best_overall}'.")

    selection = {
        "selected_on": "validation split (train fitted on train split only)",
        "effort_model": best_effort_name,
        "effort_selection_metric": f"Val {EFFORT_SELECTION_METRIC} (lower is better)",
        "effort_val_scores": {n: r["val"][EFFORT_SELECTION_METRIC] for n, r in effort_results.items()},
        "risk_family": best_risk_family,
        "risk_model": best_risk_name,
        "risk_selection_metric": f"Val {RISK_SELECTION_METRIC} (higher is better), among '+ Pred Effort' variants",
        "risk_val_scores": {n: r["val"][RISK_SELECTION_METRIC] for n, r in risk_results.items()},
        "best_risk_variant_overall_on_val": best_overall,
    }
    save_selection(selection)
    print(f"   Saved selection to {SELECTION_FILE} (used by evaluate_test_set.py and finalize_models.py)")

    # -------------------------------------------------------------------------
    # PART 3: SHAP EXPLAINABILITY for the selected risk model (validation rows)
    # -------------------------------------------------------------------------
    print("\n" + "#" * 80)
    print("3. MODEL EXPLAINABILITY & FEATURE IMPORTANCE (SHAP ANALYSIS)")
    print("#" * 80)
    print(f"Explaining: {best_risk_name}")

    pipe = fitted_risk_pipes[best_risk_name]
    prep, model = pipe.named_steps["preprocessor"], pipe.named_steps["model"]
    shap_vals = shap_values_positive_class(model, prep.transform(train_done), prep.transform(val_done))
    importance_df = pd.DataFrame({
        "Feature": prep.get_feature_names_out(),
        "Mean_Abs_SHAP": np.abs(shap_vals).mean(axis=0),
    }).sort_values("Mean_Abs_SHAP", ascending=False).reset_index(drop=True)

    print("\nTop 12 Most Influential Features Driving Delivery Risk (SHAP, log-odds):")
    print(importance_df.head(12).to_string(index=False))

    os.makedirs("reports", exist_ok=True)
    importance_df.to_csv("reports/risk_feature_importance.csv", index=False)
    with open("reports/advanced_models_results.json", "w") as f:
        json.dump({
            "effort_results": effort_results,
            "risk_results": risk_results,
            "best_effort_model": best_effort_name,
            "best_risk_model": best_risk_name,
            "top_shap_features": importance_df.head(15).to_dict(orient="records"),
        }, f, indent=2)

    print("\nSaved artifacts:")
    print(" - reports/model_selection.json")
    print(" - reports/risk_feature_importance.csv")
    print(" - reports/advanced_models_results.json")
    print("=" * 80)


if __name__ == "__main__":
    main()
