import os
import json
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold

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


def main():
    print("=" * 80)
    print("PHASE 4 & 5: ADVANCED MODELING (NLP, TWO-STAGE RISK & SHAP) - CORRECTED")
    print("=" * 80)

    df = pd.read_csv("data/processed/xd_modeling_dataset.csv")
    df["full_text"] = df["summary"].fillna("") + " " + df["description"].fillna("")

    cat_features = ["issuetype", "priority"]

    # FIX 1: Stage 1 (effort) = creation-time-only features. No early-window changelog
    # signals here - those don't exist yet at the moment Story Points are estimated.
    stage1_num_features = [
        "num_components", "num_labels", "summary_len", "description_len",
        "reporter_prev_issue_count", "priority_ordinal",
        "has_description", "has_components", "has_labels",
    ]
    # Stage 2 (risk) legitimately adds the early-window signals - that's the point
    # of observing issues at the EARLY_WINDOW_DAYS checkpoint.
    stage2_base_num_features = stage1_num_features + [
        "early_status_changes", "early_assignee_changes", "early_priority_changes",
        "early_comment_count", "had_early_status_change", "first_status_change_days_imputed",
    ]

    train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
    val_df = df[df["split"] == "val"].copy().reset_index(drop=True)

    # -------------------------------------------------------------------------
    # PART 1: EFFORT ESTIMATION WITH NLP (TF-IDF), trained on log1p(story_points)
    # -------------------------------------------------------------------------
    print("\n" + "#" * 80)
    print("1. EFFORT PREDICTION: TABULAR vs TABULAR + TF-IDF (log1p target)")
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

    y_train_eff_raw = train_df["story_points"].values
    y_train_eff_log = np.log1p(y_train_eff_raw)  # FIX 2: log1p target
    y_val_eff_raw = val_df["story_points"].values

    print(f"Story Points skew (raw): {pd.Series(y_train_eff_raw).skew():.2f} -> training on log1p")

    effort_models = {
        "Ridge (Tabular Only)": (preprocessor_tab_only, Ridge(alpha=10.0, random_state=42)),
        "Ridge (Tabular + TF-IDF)": (preprocessor_with_text, Ridge(alpha=10.0, random_state=42)),
        "LightGBM (Tabular Only)": (preprocessor_tab_only, LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, verbose=-1)),
        "LightGBM (Tabular + TF-IDF)": (preprocessor_with_text, LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, verbose=-1)),
    }

    effort_results = {}
    fitted_effort_pipelines = {}

    for name, (prep, model) in effort_models.items():
        pipe = Pipeline([("preprocessor", prep), ("model", model)])
        pipe.fit(train_df, y_train_eff_log)
        fitted_effort_pipelines[name] = pipe

        train_pred = np.expm1(pipe.predict(train_df)).clip(min=0)
        val_pred = np.expm1(pipe.predict(val_df)).clip(min=0)

        effort_results[name] = {
            "train": eval_effort_metrics(y_train_eff_raw, train_pred),
            "val": eval_effort_metrics(y_val_eff_raw, val_pred),
        }

    effort_table = pd.DataFrame([
        {"Model": name, "Val MAE": r["val"]["MAE"], "Val RMSE": r["val"]["RMSE"],
         "Val MdAE": r["val"]["MdAE"], "Val R2": r["val"]["R2"],
         "Val PRED(25)%": r["val"]["PRED(25)%"], "Val MdMRE%": r["val"]["MdMRE%"],
         "Train R2": r["train"]["R2"]}
        for name, r in effort_results.items()
    ])
    print("\nEffort Regression Results Comparison (Validation Set, log1p-trained):")
    print(effort_table.to_string(index=False))

    # FIX 3: pick the effort model that actually performed BEST on validation,
    # instead of hardcoding LightGBM+TF-IDF by assumption.
    best_effort_name = effort_table.sort_values("Val MAE").iloc[0]["Model"]
    best_prep, best_model_template = effort_models[best_effort_name]
    print(f"\n>> Selected '{best_effort_name}' as the Stage 1 model to feed Stage 2 (lowest Val MAE)")

    # -------------------------------------------------------------------------
    # PART 2: TWO-STAGE CONDITIONING: EFFORT -> RISK (leak-safe population, log target)
    # -------------------------------------------------------------------------
    print("\n" + "#" * 80)
    print("2. TWO-STAGE MODELING: FEEDING PREDICTED EFFORT INTO RISK CLASSIFIER")
    print("#" * 80)

    from sklearn.base import clone

    # Out-of-fold predicted effort for TRAIN rows (never let a row see its own model)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_train_effort_pred = np.zeros(len(train_df))
    for tr_idx, te_idx in kf.split(train_df):
        tr_fold = train_df.iloc[tr_idx]
        te_fold = train_df.iloc[te_idx]
        fold_pipe = Pipeline([("preprocessor", clone(best_prep)), ("model", clone(best_model_template))])
        fold_pipe.fit(tr_fold, np.log1p(tr_fold["story_points"].values))
        oof_train_effort_pred[te_idx] = np.expm1(fold_pipe.predict(te_fold)).clip(min=0)
    train_df["pred_effort"] = oof_train_effort_pred

    # For validation: fit the selected model on the FULL train set, predict val
    full_effort_pipe = Pipeline([("preprocessor", clone(best_prep)), ("model", clone(best_model_template))])
    full_effort_pipe.fit(train_df, np.log1p(train_df["story_points"].values))
    val_df["pred_effort"] = np.expm1(full_effort_pipe.predict(val_df)).clip(min=0)

    # FIX 4: restrict to issues still open past the observation window
    train_done = train_df[(train_df["risk_25pct"].notna()) & (train_df["eligible_for_risk_model"] == 1)].copy().reset_index(drop=True)
    val_done = val_df[(val_df["risk_25pct"].notna()) & (val_df["eligible_for_risk_model"] == 1)].copy().reset_index(drop=True)
    train_done["risk_25pct"] = train_done["risk_25pct"].astype(int)
    val_done["risk_25pct"] = val_done["risk_25pct"].astype(int)

    print(f"Risk-eligible Train: {len(train_done)} (At-risk: {train_done['risk_25pct'].mean():.1%})")
    print(f"Risk-eligible Val:   {len(val_done)} (At-risk: {val_done['risk_25pct'].mean():.1%})")

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

    # FIX 5: COMPLETE the ablation - every model type gets a without-effort AND
    # a with-effort version, not just Logistic Regression. This is the table your
    # novelty claim actually rests on.
    risk_model_specs = {
        "Logistic Regression": lambda: LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        "Random Forest": lambda: RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1),
        "LightGBM": lambda: LGBMClassifier(n_estimators=100, is_unbalance=True, random_state=42, verbose=-1),
    }

    risk_results = {}
    fitted_risk_pipes = {}

    for base_name, model_fn in risk_model_specs.items():
        for variant, prep, feat_note in [
            ("No Effort", prep_risk_base, "baseline"),
            ("+ Pred Effort", prep_risk_two_stage, "two-stage"),
        ]:
            full_name = f"{base_name} ({variant})"
            pipe = Pipeline([("preprocessor", prep), ("model", model_fn())])
            pipe.fit(train_done, train_done["risk_25pct"].values)
            fitted_risk_pipes[full_name] = pipe

            val_pred = pipe.predict(val_done)
            val_prob = pipe.predict_proba(val_done)[:, 1]
            train_pred = pipe.predict(train_done)
            train_prob = pipe.predict_proba(train_done)[:, 1]

            risk_results[full_name] = {
                "train": eval_risk_metrics(train_done["risk_25pct"].values, train_pred, train_prob),
                "val": eval_risk_metrics(val_done["risk_25pct"].values, val_pred, val_prob),
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

    print("\n--- Ablation deltas (ROC-AUC gained by adding predicted effort) ---")
    for base_name in risk_model_specs:
        no_eff = risk_results[f"{base_name} (No Effort)"]["val"]["ROC-AUC"]
        with_eff = risk_results[f"{base_name} (+ Pred Effort)"]["val"]["ROC-AUC"]
        print(f"  {base_name}: {no_eff:.4f} -> {with_eff:.4f}  (delta {with_eff - no_eff:+.4f})")

    # -------------------------------------------------------------------------
    # PART 3: SHAP EXPLAINABILITY
    # -------------------------------------------------------------------------
    print("\n" + "#" * 80)
    print("3. MODEL EXPLAINABILITY & FEATURE IMPORTANCE (SHAP ANALYSIS)")
    print("#" * 80)

    best_risk_name = risk_table.sort_values("Val ROC-AUC", ascending=False).iloc[0]["Model"]
    print(f"Explaining: {best_risk_name}")
    best_risk_pipe = fitted_risk_pipes[best_risk_name]
    prep = best_risk_pipe.named_steps["preprocessor"]
    model = best_risk_pipe.named_steps["model"]

    cat_encoder = prep.named_transformers_["cat"]
    cat_names = list(cat_encoder.get_feature_names_out(cat_features))
    used_num_features = num_features_with_effort if "+ Pred Effort" in best_risk_name else num_features_no_effort
    all_feature_names = cat_names + used_num_features

    X_val_trans = prep.transform(val_done)

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_val_trans)
        shap_vals_class1 = shap_values[1] if isinstance(shap_values, list) else shap_values
    except Exception:
        explainer = shap.Explainer(model, X_val_trans)
        shap_vals_class1 = explainer(X_val_trans).values

    mean_abs_shap = np.abs(shap_vals_class1).mean(axis=0)
    importance_df = pd.DataFrame({
        "Feature": all_feature_names, "Mean_Abs_SHAP": mean_abs_shap,
    }).sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)

    print("\nTop 12 Most Influential Features Driving Delivery Risk (SHAP):")
    print(importance_df.head(12).to_string(index=False))

    os.makedirs("reports", exist_ok=True)
    importance_df.to_csv("reports/risk_feature_importance.csv", index=False)

    final_output = {
        "effort_results": effort_results,
        "risk_results": risk_results,
        "best_effort_model": best_effort_name,
        "best_risk_model": best_risk_name,
        "top_shap_features": importance_df.head(15).to_dict(orient="records"),
    }
    with open("reports/advanced_models_results.json", "w") as f:
        json.dump(final_output, f, indent=2)

    print("\nSaved artifacts:")
    print(" - reports/risk_feature_importance.csv")
    print(" - reports/advanced_models_results.json")
    print("=" * 80)


if __name__ == "__main__":
    main()