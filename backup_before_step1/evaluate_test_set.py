import os
import json
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.base import clone

from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

from sklearn.metrics import (
    mean_absolute_error, root_mean_squared_error, median_absolute_error, r2_score,
    roc_auc_score, average_precision_score, precision_score, recall_score,
    f1_score, accuracy_score, brier_score_loss,
)
import shap

EARLY_WINDOW_DAYS = 5
N_BOOTSTRAP = 1000
RANDOM_SEED = 42


def eval_effort_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mdae = median_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    safe_y_true = np.where(y_true == 0, 1e-6, y_true)
    rel_errors = np.abs(y_true - y_pred) / safe_y_true
    return {
        "MAE": round(mae, 4), "RMSE": round(rmse, 4), "MdAE": round(mdae, 4),
        "R2": round(r2, 4), "PRED(25)%": round(np.mean(rel_errors <= 0.25) * 100.0, 2),
        "MdMRE%": round(np.median(rel_errors) * 100.0, 2),
    }


def eval_risk_metrics(y_true, y_pred, y_prob):
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = 0.5
    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except Exception:
        pr_auc = np.mean(y_true)
    return {
        "ROC-AUC": round(roc_auc, 4), "PR-AUC": round(pr_auc, 4),
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "AtRisk-Precision": round(precision_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "AtRisk-Recall": round(recall_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "AtRisk-F1": round(f1_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "Macro-F1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "Brier": round(brier_score_loss(y_true, y_prob), 4),
    }


def bootstrap_auc_delta(y_true, prob_no_effort, prob_with_effort, n_boot=N_BOOTSTRAP, seed=RANDOM_SEED):
    """
    Paired bootstrap on the TEST rows: resample row indices with replacement,
    recompute ROC-AUC for both variants on the SAME resampled rows each time,
    and record the delta. Both probability arrays already come from models
    fit once on Train+Val - no refitting happens here, only resampling of the
    already-computed predictions, which is what makes this fast and valid for
    a paired comparison (same examples, two models).
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    n = len(y_true)
    deltas = np.empty(n_boot)

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y_true[idx]
        if len(np.unique(yb)) < 2:
            idx = rng.integers(0, n, size=n)
            yb = y_true[idx]
            if len(np.unique(yb)) < 2:
                deltas[b] = np.nan
                continue
        auc_no = roc_auc_score(yb, prob_no_effort[idx])
        auc_with = roc_auc_score(yb, prob_with_effort[idx])
        deltas[b] = auc_with - auc_no

    deltas = deltas[~np.isnan(deltas)]
    point_estimate = roc_auc_score(y_true, prob_with_effort) - roc_auc_score(y_true, prob_no_effort)
    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
    significant = (ci_low > 0) or (ci_high < 0)
    return {
        "point_delta": round(float(point_estimate), 4),
        "ci_low_95": round(float(ci_low), 4),
        "ci_high_95": round(float(ci_high), 4),
        "significant_at_95": bool(significant),
        "n_bootstrap": int(len(deltas)),
    }


def main():
    print("=" * 80)
    print("PHASE 6 (FINAL): TEST EVALUATION + BOOTSTRAP CI + CORRECTED SHAP")
    print("=" * 80)

    df = pd.read_csv("data/processed/xd_modeling_dataset.csv")
    df["full_text"] = df["summary"].fillna("") + " " + df["description"].fillna("")

    cat_features = ["issuetype", "priority"]
    stage1_num_features = [
        "num_components", "num_labels", "summary_len", "description_len",
        "reporter_prev_issue_count", "priority_ordinal",
        "has_description", "has_components", "has_labels",
    ]
    stage2_base_num_features = stage1_num_features + [
        "early_status_changes", "early_assignee_changes", "early_priority_changes",
        "early_comment_count", "had_early_status_change", "first_status_change_days_imputed",
    ]

    train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
    val_df = df[df["split"] == "val"].copy().reset_index(drop=True)
    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)
    trainval_df = pd.concat([train_df, val_df], ignore_index=True)

    print(f"Data split sizes: Train={len(train_df)}, Val={len(val_df)}, "
          f"Train+Val={len(trainval_df)}, Test={len(test_df)}")

    # -------------------------------------------------------------------------
    # TASK A: EFFORT PREDICTION ON TEST SPLIT
    # -------------------------------------------------------------------------
    print("\n" + "#" * 80)
    print("TASK A: EFFORT PREDICTION FINAL TEST BENCHMARK")
    print("#" * 80)

    preprocessor_with_text = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ("num", StandardScaler(), stage1_num_features),
        ("text", TfidfVectorizer(max_features=300, stop_words="english", ngram_range=(1, 2), sublinear_tf=True), "full_text"),
    ])
    preprocessor_tab_only = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ("num", StandardScaler(), stage1_num_features),
    ])

    y_trainval_eff_raw = trainval_df["story_points"].values
    y_trainval_eff_log = np.log1p(y_trainval_eff_raw)
    y_test_eff_raw = test_df["story_points"].values

    effort_models = {
        "Dummy (Median)": (preprocessor_tab_only, DummyRegressor(strategy="median")),
        "Ridge (Tabular Only)": (preprocessor_tab_only, Ridge(alpha=10.0, random_state=42)),
        "Ridge (Tabular + TF-IDF)": (preprocessor_with_text, Ridge(alpha=10.0, random_state=42)),
        "LightGBM (Tabular + TF-IDF)": (preprocessor_with_text, LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, verbose=-1)),
    }

    effort_test_results = {}
    for name, (prep, model) in effort_models.items():
        pipe = Pipeline([("preprocessor", prep), ("model", model)])
        pipe.fit(trainval_df, y_trainval_eff_log)
        test_pred = np.expm1(pipe.predict(test_df)).clip(min=0)
        effort_test_results[name] = eval_effort_metrics(y_test_eff_raw, test_pred)

    effort_test_table = pd.DataFrame([
        {"Model": name, "Test MAE": r["MAE"], "Test RMSE": r["RMSE"], "Test MdAE": r["MdAE"],
         "Test R2": r["R2"], "Test PRED(25)%": r["PRED(25)%"], "Test MdMRE%": r["MdMRE%"]}
        for name, r in effort_test_results.items()
    ])
    print("\nEffort Prediction Results on Held-Out Test Set (fit on Train+Val):")
    print(effort_test_table.to_string(index=False))

    best_effort_name = effort_test_table[effort_test_table["Model"] != "Dummy (Median)"].sort_values("Test MAE").iloc[0]["Model"]
    best_prep, best_model_template = effort_models[best_effort_name]
    print(f"\n>> Best non-dummy effort model on test: {best_effort_name}")

    # -------------------------------------------------------------------------
    # TASK B: RISK PREDICTION - COMPLETE ABLATION ON TEST
    # -------------------------------------------------------------------------
    print("\n" + "#" * 80)
    print("TASK B: DELIVERY RISK PREDICTION FINAL TEST BENCHMARK")
    print("#" * 80)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_trainval_effort = np.zeros(len(trainval_df))
    for tr_idx, te_idx in kf.split(trainval_df):
        tr_f = trainval_df.iloc[tr_idx]
        te_f = trainval_df.iloc[te_idx]
        fold_pipe = Pipeline([("preprocessor", clone(best_prep)), ("model", clone(best_model_template))])
        fold_pipe.fit(tr_f, np.log1p(tr_f["story_points"].values))
        oof_trainval_effort[te_idx] = np.expm1(fold_pipe.predict(te_f)).clip(min=0)
    trainval_df["pred_effort"] = oof_trainval_effort

    full_effort_pipe = Pipeline([("preprocessor", clone(best_prep)), ("model", clone(best_model_template))])
    full_effort_pipe.fit(trainval_df, np.log1p(trainval_df["story_points"].values))
    test_df["pred_effort"] = np.expm1(full_effort_pipe.predict(test_df)).clip(min=0)

    trainval_done = trainval_df[(trainval_df["risk_25pct"].notna()) & (trainval_df["eligible_for_risk_model"] == 1)].copy().reset_index(drop=True)
    test_done = test_df[(test_df["risk_25pct"].notna()) & (test_df["eligible_for_risk_model"] == 1)].copy().reset_index(drop=True)
    trainval_done["risk_25pct"] = trainval_done["risk_25pct"].astype(int)
    test_done["risk_25pct"] = test_done["risk_25pct"].astype(int)

    print(f"Risk-eligible Train+Val: {len(trainval_done)} (At-risk: {trainval_done['risk_25pct'].mean():.1%})")
    print(f"Risk-eligible Test:      {len(test_done)} (At-risk: {test_done['risk_25pct'].mean():.1%})")

    num_features_no_effort = stage2_base_num_features
    num_features_with_effort = stage2_base_num_features + ["pred_effort"]

    prep_risk_base = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ("num", StandardScaler(), num_features_no_effort),
    ])
    prep_risk_two_stage = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ("num", StandardScaler(), num_features_with_effort),
    ])

    risk_model_fns = {
        "Logistic Regression": lambda: LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        "Random Forest": lambda: RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1),
        "LightGBM": lambda: LGBMClassifier(n_estimators=100, is_unbalance=True, random_state=42, verbose=-1),
    }

    risk_test_results = {}
    fitted_risk_pipes = {}
    test_probs = {}  # full_name -> array of P(at risk) on test_done, for bootstrap CI

    dummy_pipe = Pipeline([("preprocessor", prep_risk_base), ("model", DummyClassifier(strategy="prior"))])
    dummy_pipe.fit(trainval_done, trainval_done["risk_25pct"].values)
    dp = dummy_pipe.predict(test_done)
    dpr = dummy_pipe.predict_proba(test_done)[:, 1]
    risk_test_results["Dummy (Majority Class)"] = eval_risk_metrics(test_done["risk_25pct"].values, dp, dpr)
    fitted_risk_pipes["Dummy (Majority Class)"] = dummy_pipe
    test_probs["Dummy (Majority Class)"] = dpr

    for base_name, model_fn in risk_model_fns.items():
        for variant, prep in [("No Effort", prep_risk_base), ("+ Pred Effort", prep_risk_two_stage)]:
            full_name = f"{base_name} ({variant})"
            pipe = Pipeline([("preprocessor", prep), ("model", model_fn())])
            pipe.fit(trainval_done, trainval_done["risk_25pct"].values)
            fitted_risk_pipes[full_name] = pipe
            test_pred = pipe.predict(test_done)
            test_prob = pipe.predict_proba(test_done)[:, 1]
            risk_test_results[full_name] = eval_risk_metrics(test_done["risk_25pct"].values, test_pred, test_prob)
            test_probs[full_name] = test_prob

    risk_test_table = pd.DataFrame([
        {"Model": name, "Test ROC-AUC": r["ROC-AUC"], "Test PR-AUC": r["PR-AUC"],
         "Test Macro-F1": r["Macro-F1"], "Test AtRisk-Recall": r["AtRisk-Recall"],
         "Test AtRisk-Precision": r["AtRisk-Precision"], "Test AtRisk-F1": r["AtRisk-F1"],
         "Test Brier": r["Brier"]}
        for name, r in risk_test_results.items()
    ])
    print("\nRisk Prediction Results on Held-Out Test Set (COMPLETE ablation, leak-safe population):")
    print(risk_test_table.to_string(index=False))

    # -------------------------------------------------------------------------
    # BOOTSTRAP CONFIDENCE INTERVALS ON THE ABLATION DELTAS
    # -------------------------------------------------------------------------
    print("\n" + "#" * 80)
    print(f"BOOTSTRAP 95% CONFIDENCE INTERVALS ON ABLATION DELTAS (n={N_BOOTSTRAP} resamples)")
    print("#" * 80)

    y_test_risk = test_done["risk_25pct"].values
    bootstrap_results = {}
    for base_name in risk_model_fns:
        prob_no = test_probs[f"{base_name} (No Effort)"]
        prob_with = test_probs[f"{base_name} (+ Pred Effort)"]
        res = bootstrap_auc_delta(y_test_risk, prob_no, prob_with)
        bootstrap_results[base_name] = res
        sig_marker = "SIGNIFICANT (CI excludes 0)" if res["significant_at_95"] else "not significant (CI includes 0)"
        print(f"  {base_name}: delta={res['point_delta']:+.4f}  "
              f"95% CI=[{res['ci_low_95']:+.4f}, {res['ci_high_95']:+.4f}]  -> {sig_marker}")

    # -------------------------------------------------------------------------
    # SHAP - explain the model actually reported as the final Test result
    # -------------------------------------------------------------------------
    print("\n" + "#" * 80)
    print("MODEL EXPLAINABILITY (SHAP) - ON THE ACTUAL TEST-BEST MODEL")
    print("#" * 80)

    real_model_rows = risk_test_table[risk_test_table["Model"] != "Dummy (Majority Class)"]
    best_risk_name = real_model_rows.sort_values("Test ROC-AUC", ascending=False).iloc[0]["Model"]
    print(f"Test-best model (by Test ROC-AUC): {best_risk_name}")
    print("(This replaces the earlier version, which explained the Val-best Logistic")
    print(" Regression model even though it was not the model with the best Test score.)")

    best_risk_pipe = fitted_risk_pipes[best_risk_name]
    prep = best_risk_pipe.named_steps["preprocessor"]
    model = best_risk_pipe.named_steps["model"]

    cat_encoder = prep.named_transformers_["cat"]
    cat_names = list(cat_encoder.get_feature_names_out(cat_features))
    used_num_features = num_features_with_effort if "+ Pred Effort" in best_risk_name else num_features_no_effort
    all_feature_names = cat_names + used_num_features

    X_test_trans = prep.transform(test_done)
    if hasattr(X_test_trans, "toarray"):
        X_test_trans = X_test_trans.toarray()

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_trans)
        shap_vals_class1 = shap_values[1] if isinstance(shap_values, list) else shap_values
    except Exception:
        explainer = shap.Explainer(model, X_test_trans)
        shap_vals_class1 = explainer(X_test_trans).values

    mean_abs_shap = np.abs(shap_vals_class1).mean(axis=0)
    importance_df = pd.DataFrame({
        "Feature": all_feature_names, "Mean_Abs_SHAP": mean_abs_shap,
    }).sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)

    print(f"\nTop 12 Most Influential Features Driving Delivery Risk ({best_risk_name}, evaluated on Test):")
    print(importance_df.head(12).to_string(index=False))

    # -------------------------------------------------------------------------
    # SAVE EVERYTHING
    # -------------------------------------------------------------------------
    os.makedirs("reports", exist_ok=True)
    importance_df.to_csv("reports/risk_feature_importance_test.csv", index=False)

    final_output = {
        "effort_test_results": effort_test_results,
        "risk_test_results": risk_test_results,
        "best_effort_model": best_effort_name,
        "best_risk_model_test": best_risk_name,
        "bootstrap_ablation_ci": bootstrap_results,
        "top_shap_features_test": importance_df.head(15).to_dict(orient="records"),
    }
    with open("reports/test_evaluation_results.json", "w") as f:
        json.dump(final_output, f, indent=2)

    print("\nSaved:")
    print(" - reports/test_evaluation_results.json  (now includes bootstrap CIs + Test-matched SHAP)")
    print(" - reports/risk_feature_importance_test.csv")
    print("=" * 80)


if __name__ == "__main__":
    main()