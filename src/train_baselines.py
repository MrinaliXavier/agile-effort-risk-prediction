import json
import os

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline

from common import (CAT_FEATURES, EARLY_WINDOW_DAYS, RANDOM_SEED, RISK_LABEL,
                    STAGE1_NUM_FEATURES, STAGE2_BASE_NUM_FEATURES, build_preprocessor,
                    eval_effort_metrics, eval_risk_metrics, risk_population)


def main():
    print("=" * 80)
    print("PHASE 3: BASELINE MODEL TRAINING & BENCHMARKING")
    print("=" * 80)

    df = pd.read_csv("data/processed/xd_modeling_dataset.csv")
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]

    # -------------------------------------------------------------------------
    # TASK A: EFFORT PREDICTION (REGRESSION) - target: log1p(story_points)
    # Stage 1 uses only information available when the ticket is created.
    # -------------------------------------------------------------------------
    print("\n" + "#" * 80)
    print("TASK A: EFFORT PREDICTION (REGRESSION: Story Points, log1p target)")
    print("#" * 80)

    X_train_eff = train_df[CAT_FEATURES + STAGE1_NUM_FEATURES]
    y_train_eff_raw = train_df["story_points"].values
    y_train_eff_log = np.log1p(y_train_eff_raw)
    X_val_eff = val_df[CAT_FEATURES + STAGE1_NUM_FEATURES]
    y_val_eff_raw = val_df["story_points"].values

    print(f"Effort Train samples: {len(X_train_eff)} | Val samples: {len(X_val_eff)}")
    print(f"Story Points skew (raw): {pd.Series(y_train_eff_raw).skew():.2f} -> training on log1p")

    effort_models = {
        "Dummy (Median)": DummyRegressor(strategy="median"),
        "Ridge Regression": Ridge(alpha=1.0, random_state=RANDOM_SEED),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_SEED),
    }

    effort_results = {}
    for name, model in effort_models.items():
        pipe = Pipeline([("preprocessor", build_preprocessor(STAGE1_NUM_FEATURES)), ("model", model)])
        pipe.fit(X_train_eff, y_train_eff_log)
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

    done_df = risk_population(df)
    print(f"Risk-eligible population (excludes issues resolved within {EARLY_WINDOW_DAYS} days): {len(done_df)}")

    train_done = done_df[done_df["split"] == "train"]
    val_done = done_df[done_df["split"] == "val"]
    X_train_risk = train_done[CAT_FEATURES + STAGE2_BASE_NUM_FEATURES]
    y_train_risk = train_done[RISK_LABEL].values
    X_val_risk = val_done[CAT_FEATURES + STAGE2_BASE_NUM_FEATURES]
    y_val_risk = val_done[RISK_LABEL].values

    print(f"Risk Train samples: {len(X_train_risk)} (At-risk: {y_train_risk.mean():.1%})")
    print(f"Risk Val samples: {len(X_val_risk)} (At-risk: {y_val_risk.mean():.1%})")

    risk_models = {
        "Dummy (Majority Class)": DummyClassifier(strategy="prior"),
        "Logistic Regression (Balanced)": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED),
        "Random Forest (Balanced)": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_SEED),
    }

    risk_results = {}
    for name, model in risk_models.items():
        pipe = Pipeline([("preprocessor", build_preprocessor(STAGE2_BASE_NUM_FEATURES)), ("model", model)])
        pipe.fit(X_train_risk, y_train_risk)
        risk_results[name] = {
            "train": eval_risk_metrics(y_train_risk, pipe.predict(X_train_risk), pipe.predict_proba(X_train_risk)[:, 1]),
            "val": eval_risk_metrics(y_val_risk, pipe.predict(X_val_risk), pipe.predict_proba(X_val_risk)[:, 1]),
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
