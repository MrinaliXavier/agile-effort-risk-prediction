import os
import json
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import (
    mean_absolute_error, root_mean_squared_error, median_absolute_error, r2_score,
    roc_auc_score, average_precision_score, precision_score, recall_score,
    f1_score, accuracy_score, brier_score_loss,
)

EARLY_WINDOW_DAYS = 5


def eval_effort_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mdae = median_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    safe_y_true = np.where(y_true == 0, 1e-6, y_true)
    rel_errors = np.abs(y_true - y_pred) / safe_y_true
    pred_25 = np.mean(rel_errors <= 0.25) * 100.0
    mdmre = np.median(rel_errors) * 100.0

    return {
        "MAE": round(mae, 4), "RMSE": round(rmse, 4), "MdAE": round(mdae, 4),
        "R2": round(r2, 4), "PRED(25)%": round(pred_25, 2), "MdMRE%": round(mdmre, 2),
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
        "ROC-AUC": round(roc_auc, 4),
        "PR-AUC": round(pr_auc, 4),
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "AtRisk-Precision": round(precision_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "AtRisk-Recall": round(recall_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "AtRisk-F1": round(f1_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "Macro-F1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "Brier": round(brier_score_loss(y_true, y_prob), 4),
    }


def main():
    print("=" * 80)
    print("PHASE 3: BASELINE MODEL TRAINING & BENCHMARKING (CORRECTED)")
    print("=" * 80)

    df = pd.read_csv("data/processed/xd_modeling_dataset.csv")

    cat_features = ["issuetype", "priority"]

    # FIX 1: Stage 1 (effort) must only use information knowable AT issue creation.
    # The early-window changelog features (early_status_changes, first_status_change_days,
    # etc.) require waiting EARLY_WINDOW_DAYS after creation to exist - using them to
    # predict Story Points (an upfront estimate) is look-ahead leakage for this task.
    stage1_num_features = [
        "num_components", "num_labels", "summary_len", "description_len",
        "reporter_prev_issue_count", "priority_ordinal",
        "has_description", "has_components", "has_labels",
    ]

    # Stage 2 (risk) legitimately observes the world at the EARLY_WINDOW_DAYS checkpoint,
    # so the early-window features are valid here - that's the whole point of the 5-day
    # observation window.
    stage2_num_features = stage1_num_features + [
        "early_status_changes", "early_assignee_changes", "early_priority_changes",
        "early_comment_count", "had_early_status_change", "first_status_change_days_imputed",
    ]

    # -------------------------------------------------------------------------
    # TASK A: EFFORT PREDICTION (REGRESSION) - target: log1p(story_points)
    # -------------------------------------------------------------------------
    print("\n" + "#" * 80)
    print("TASK A: EFFORT PREDICTION (REGRESSION: Story Points, log1p target)")
    print("#" * 80)

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]

    stage1_preprocessor = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ("num", StandardScaler(), stage1_num_features),
    ])

    X_train_eff = train_df[cat_features + stage1_num_features]
    y_train_eff_raw = train_df["story_points"].values
    y_train_eff_log = np.log1p(y_train_eff_raw)  # FIX 2: train on log1p scale

    X_val_eff = val_df[cat_features + stage1_num_features]
    y_val_eff_raw = val_df["story_points"].values

    print(f"Effort Train samples: {len(X_train_eff)} | Val samples: {len(X_val_eff)}")
    print(f"Story Points skew (raw): {pd.Series(y_train_eff_raw).skew():.2f} -> training on log1p to tame this")

    effort_models = {
        "Dummy (Median)": DummyRegressor(strategy="median"),
        "Ridge Regression": Ridge(alpha=1.0, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }

    effort_results = {}
    for name, model in effort_models.items():
        pipe = Pipeline([("preprocessor", stage1_preprocessor), ("model", model)])
        pipe.fit(X_train_eff, y_train_eff_log)

        # Predict in log space, invert back to the real Story Points scale before scoring
        train_pred = np.expm1(pipe.predict(X_train_eff)).clip(min=0)
        val_pred = np.expm1(pipe.predict(X_val_eff)).clip(min=0)

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
    print("\nEffort Prediction Baseline Results (Validation Set, log1p-trained):")
    print(effort_table.to_string(index=False))

    # -------------------------------------------------------------------------
    # TASK B: RISK PREDICTION (BINARY CLASSIFICATION) - leak-safe population only
    # -------------------------------------------------------------------------
    print("\n" + "#" * 80)
    print("TASK B: RISK PREDICTION (BINARY CLASSIFICATION: Delayed Cycle Time Residual)")
    print("#" * 80)

    # FIX 3: only issues still open past the observation window are genuine prediction
    # targets. Issues that already resolved within the window are 100% "on track" by
    # construction and would let the model trivially read the outcome off the changelog.
    done_df = df[(df["risk_25pct"].notna()) & (df["eligible_for_risk_model"] == 1)].copy()
    done_df["risk_25pct"] = done_df["risk_25pct"].astype(int)
    print(f"Risk-eligible population (excludes issues resolved within {EARLY_WINDOW_DAYS} days): {len(done_df)}")

    stage2_preprocessor = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ("num", StandardScaler(), stage2_num_features),
    ])

    train_done = done_df[done_df["split"] == "train"]
    val_done = done_df[done_df["split"] == "val"]

    X_train_risk = train_done[cat_features + stage2_num_features]
    y_train_risk = train_done["risk_25pct"].values
    X_val_risk = val_done[cat_features + stage2_num_features]
    y_val_risk = val_done["risk_25pct"].values

    print(f"Risk Train samples: {len(X_train_risk)} (At-risk: {y_train_risk.mean():.1%})")
    print(f"Risk Val samples: {len(X_val_risk)} (At-risk: {y_val_risk.mean():.1%})")

    risk_models = {
        "Dummy (Majority Class)": DummyClassifier(strategy="prior"),
        "Logistic Regression (Balanced)": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        "Random Forest (Balanced)": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    risk_results = {}
    for name, model in risk_models.items():
        pipe = Pipeline([("preprocessor", stage2_preprocessor), ("model", model)])
        pipe.fit(X_train_risk, y_train_risk)

        val_pred = pipe.predict(X_val_risk)
        val_prob = pipe.predict_proba(X_val_risk)[:, 1] if hasattr(pipe, "predict_proba") else val_pred
        train_pred = pipe.predict(X_train_risk)
        train_prob = pipe.predict_proba(X_train_risk)[:, 1] if hasattr(pipe, "predict_proba") else train_pred

        risk_results[name] = {
            "train": eval_risk_metrics(y_train_risk, train_pred, train_prob),
            "val": eval_risk_metrics(y_val_risk, val_pred, val_prob),
        }

    risk_table = pd.DataFrame([
        {"Model": name, "ROC-AUC": r["val"]["ROC-AUC"], "PR-AUC": r["val"]["PR-AUC"],
         "Macro-F1": r["val"]["Macro-F1"], "AtRisk-Recall": r["val"]["AtRisk-Recall"],
         "AtRisk-Precision": r["val"]["AtRisk-Precision"], "AtRisk-F1": r["val"]["AtRisk-F1"],
         "Brier": r["val"]["Brier"]}
        for name, r in risk_results.items()
    ])
    print("\nRisk Prediction Baseline Results (Validation Set, leak-safe population):")
    print(risk_table.to_string(index=False))

    os.makedirs("reports", exist_ok=True)
    with open("reports/baseline_results.json", "w") as f:
        json.dump({"effort_results": effort_results, "risk_results": risk_results}, f, indent=2)
    print("\nSaved baseline evaluation results to: reports/baseline_results.json")
    print("=" * 80)


if __name__ == "__main__":
    main()