"""
PHASE 6 (FINAL): held-out TEST evaluation of the models selected on VALIDATION.

No model is chosen here. The selected models are read from
reports/model_selection.json (written by train_advanced_models.py). All other
candidates are still scored on test and reported for transparency, but they are
labelled "not selected" and play no part in the final claim.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from common import (EFFORT_CANDIDATES, RANDOM_SEED, RISK_CANDIDATES, RISK_LABEL, RISK_VARIANTS,
                    STAGE1_NUM_FEATURES, STAGE2_BASE_NUM_FEATURES, add_full_text,
                    build_effort_pipeline, build_preprocessor, build_risk_pipeline,
                    eval_effort_metrics, eval_risk_metrics, load_selection,
                    oof_effort_predictions, predict_effort, risk_name, risk_population,
                    shap_values_positive_class)

N_BOOTSTRAP = 1000


def bootstrap_auc_delta(y_true, prob_no_effort, prob_with_effort, n_boot=N_BOOTSTRAP, seed=RANDOM_SEED):
    """
    Paired bootstrap on the TEST rows: resample row indices with replacement,
    recompute ROC-AUC for both variants on the SAME resampled rows each time,
    and record the delta. No refitting - only the already-computed predictions
    are resampled, which makes this a valid paired comparison (same rows, two models).
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    n = len(y_true)
    deltas = []
    while len(deltas) < n_boot:
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        deltas.append(roc_auc_score(y_true[idx], prob_with_effort[idx]) -
                      roc_auc_score(y_true[idx], prob_no_effort[idx]))
    deltas = np.array(deltas)
    point = roc_auc_score(y_true, prob_with_effort) - roc_auc_score(y_true, prob_no_effort)
    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
    return {
        "point_delta": round(float(point), 4),
        "ci_low_95": round(float(ci_low), 4),
        "ci_high_95": round(float(ci_high), 4),
        "significant_at_95": bool(ci_low > 0 or ci_high < 0),
        "n_bootstrap": int(len(deltas)),
    }


def main():
    print("=" * 80)
    print("PHASE 6 (FINAL): TEST EVALUATION OF VALIDATION-SELECTED MODELS")
    print("=" * 80)

    selection = load_selection()
    effort_name = selection["effort_model"]
    risk_family = selection["risk_family"]
    selected_risk = selection["risk_model"]
    print(f"Selected on validation -> Stage 1: {effort_name} | Stage 2: {selected_risk}")
    print("(Test results below are reported for these models; nothing is re-selected on test.)")

    df = add_full_text(pd.read_csv("data/processed/xd_modeling_dataset.csv"))
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)
    trainval_df = pd.concat([train_df, val_df], ignore_index=True)
    print(f"Data split sizes: Train={len(train_df)}, Val={len(val_df)}, "
          f"Train+Val={len(trainval_df)}, Test={len(test_df)}")

    # -------------------------------------------------------------------------
    # TASK A: EFFORT - every candidate refit on Train+Val, scored once on Test
    # -------------------------------------------------------------------------
    print("\n" + "#" * 80)
    print("TASK A: EFFORT PREDICTION - HELD-OUT TEST")
    print("#" * 80)

    y_tv_log = np.log1p(trainval_df["story_points"].values)
    y_test = test_df["story_points"].values

    effort_test_results = {}
    dummy = Pipeline([("preprocessor", build_preprocessor(STAGE1_NUM_FEATURES)),
                      ("model", DummyRegressor(strategy="median"))]).fit(trainval_df, y_tv_log)
    effort_test_results["Dummy (Median)"] = eval_effort_metrics(y_test, predict_effort(dummy, test_df))
    for name in EFFORT_CANDIDATES:
        pipe = build_effort_pipeline(name).fit(trainval_df, y_tv_log)
        effort_test_results[name] = eval_effort_metrics(y_test, predict_effort(pipe, test_df))

    effort_test_table = pd.DataFrame([
        {"Model": name, "Selected": "YES" if name == effort_name else "",
         "Test MAE": r["MAE"], "Test RMSE": r["RMSE"], "Test MdAE": r["MdAE"],
         "Test R2": r["R2"], "Test PRED(25)%": r["PRED(25)%"], "Test MdMRE%": r["MdMRE%"]}
        for name, r in effort_test_results.items()
    ])
    print("\nEffort Prediction Results on Held-Out Test Set (fit on Train+Val):")
    print(effort_test_table.to_string(index=False))

    # -------------------------------------------------------------------------
    # TASK B: RISK - complete ablation, predicted effort from the SELECTED effort model
    # -------------------------------------------------------------------------
    print("\n" + "#" * 80)
    print("TASK B: DELIVERY RISK PREDICTION - HELD-OUT TEST")
    print("#" * 80)

    trainval_df["pred_effort"] = oof_effort_predictions(trainval_df, effort_name)
    full_effort_pipe = build_effort_pipeline(effort_name).fit(trainval_df, y_tv_log)
    test_df["pred_effort"] = predict_effort(full_effort_pipe, test_df)

    trainval_done = risk_population(trainval_df)
    test_done = risk_population(test_df)
    y_tv_risk = trainval_done[RISK_LABEL].values
    y_test_risk = test_done[RISK_LABEL].values
    print(f"Risk-eligible Train+Val: {len(trainval_done)} (At-risk: {y_tv_risk.mean():.1%})")
    print(f"Risk-eligible Test:      {len(test_done)} (At-risk: {y_test_risk.mean():.1%})")

    risk_test_results, fitted_risk_pipes, test_probs = {}, {}, {}

    dummy_pipe = Pipeline([("preprocessor", build_preprocessor(STAGE2_BASE_NUM_FEATURES)),
                           ("model", DummyClassifier(strategy="prior"))]).fit(trainval_done, y_tv_risk)
    risk_test_results["Dummy (Majority Class)"] = eval_risk_metrics(
        y_test_risk, dummy_pipe.predict(test_done), dummy_pipe.predict_proba(test_done)[:, 1])

    for family in RISK_CANDIDATES:
        for variant in RISK_VARIANTS:
            name = risk_name(family, variant)
            pipe = build_risk_pipeline(family, with_effort=(variant == "+ Pred Effort"))
            pipe.fit(trainval_done, y_tv_risk)
            prob = pipe.predict_proba(test_done)[:, 1]
            fitted_risk_pipes[name], test_probs[name] = pipe, prob
            risk_test_results[name] = eval_risk_metrics(y_test_risk, pipe.predict(test_done), prob)

    risk_test_table = pd.DataFrame([
        {"Model": name, "Selected": "YES" if name == selected_risk else "",
         "Test ROC-AUC": r["ROC-AUC"], "Test PR-AUC": r["PR-AUC"],
         "Test Macro-F1": r["Macro-F1"], "Test AtRisk-Recall": r["AtRisk-Recall"],
         "Test AtRisk-Precision": r["AtRisk-Precision"], "Test AtRisk-F1": r["AtRisk-F1"],
         "Test Brier": r["Brier"]}
        for name, r in risk_test_results.items()
    ])
    print("\nRisk Prediction Results on Held-Out Test Set (complete ablation, leak-safe population):")
    print(risk_test_table.to_string(index=False))

    # -------------------------------------------------------------------------
    # BOOTSTRAP CONFIDENCE INTERVALS ON THE ABLATION DELTAS
    # -------------------------------------------------------------------------
    print("\n" + "#" * 80)
    print(f"BOOTSTRAP 95% CONFIDENCE INTERVALS ON ABLATION DELTAS (n={N_BOOTSTRAP} resamples)")
    print("#" * 80)

    bootstrap_results = {}
    for family in RISK_CANDIDATES:
        res = bootstrap_auc_delta(y_test_risk, test_probs[risk_name(family, "No Effort")],
                                  test_probs[risk_name(family, "+ Pred Effort")])
        bootstrap_results[family] = res
        tag = "  <- SELECTED (primary result)" if family == risk_family else "  (secondary)"
        sig = "SIGNIFICANT (CI excludes 0)" if res["significant_at_95"] else "not significant (CI includes 0)"
        print(f"  {family}: delta={res['point_delta']:+.4f}  95% CI=[{res['ci_low_95']:+.4f}, "
              f"{res['ci_high_95']:+.4f}]  -> {sig}{tag}")

    # -------------------------------------------------------------------------
    # SHAP - explain the SELECTED risk model on the test rows
    # -------------------------------------------------------------------------
    print("\n" + "#" * 80)
    print(f"MODEL EXPLAINABILITY (SHAP) - SELECTED MODEL: {selected_risk}")
    print("#" * 80)

    pipe = fitted_risk_pipes[selected_risk]
    prep, model = pipe.named_steps["preprocessor"], pipe.named_steps["model"]
    shap_vals = shap_values_positive_class(model, prep.transform(trainval_done), prep.transform(test_done))
    importance_df = pd.DataFrame({
        "Feature": prep.get_feature_names_out(),
        "Mean_Abs_SHAP": np.abs(shap_vals).mean(axis=0),
    }).sort_values("Mean_Abs_SHAP", ascending=False).reset_index(drop=True)
    print("\nTop 12 Most Influential Features Driving Delivery Risk (evaluated on Test, log-odds):")
    print(importance_df.head(12).to_string(index=False))

    os.makedirs("reports", exist_ok=True)
    importance_df.to_csv("reports/risk_feature_importance_test.csv", index=False)
    with open("reports/test_evaluation_results.json", "w") as f:
        json.dump({
            "selection_used": selection,
            "effort_test_results": effort_test_results,
            "risk_test_results": risk_test_results,
            "selected_effort_model": effort_name,
            "selected_risk_model": selected_risk,
            "bootstrap_ablation_ci": bootstrap_results,
            "top_shap_features_test": importance_df.head(15).to_dict(orient="records"),
        }, f, indent=2)

    print("\nSaved:")
    print(" - reports/test_evaluation_results.json")
    print(" - reports/risk_feature_importance_test.csv")
    print("=" * 80)


if __name__ == "__main__":
    main()
