"""
common.py - one shared definition of features, models, metrics and model selection.

Every training / evaluation / inference script imports from here, so the feature
lists and model settings can never drift apart between scripts again.
"""
import json
import os

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, f1_score,
    mean_absolute_error, median_absolute_error, precision_score, r2_score,
    recall_score, roc_auc_score, root_mean_squared_error,
)
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

EARLY_WINDOW_DAYS = 5
RANDOM_SEED = 42
RISK_LABEL = "risk_25pct"
SELECTION_FILE = "reports/model_selection.json"

# -----------------------------------------------------------------------------
# FEATURES
# -----------------------------------------------------------------------------
CAT_FEATURES = ["issuetype", "priority"]
# Reference ("baseline") category for each categorical feature = its most common
# value in the training split. One-hot columns are created for every OTHER
# category, so each coefficient reads as "difference vs a Minor-priority Story".
# Without dropping one category, the one-hot columns always sum to 1 and are
# perfectly collinear with the intercept, which makes coefficients unidentifiable.
CAT_BASELINES = ["Story", "Minor"]

# Stage 1 (effort): only information available when the ticket is created.
STAGE1_NUM_FEATURES = [
    "num_components", "num_labels", "summary_len", "description_len",
    "reporter_prev_issue_count", "has_description",
]

# Stage 2 (risk): adds activity observed during the first EARLY_WINDOW_DAYS.
STAGE2_BASE_NUM_FEATURES = STAGE1_NUM_FEATURES + [
    "early_status_changes", "early_assignee_changes", "early_priority_changes",
    "early_comment_count", "first_status_change_days_imputed",
]
STAGE2_NUM_FEATURES_WITH_EFFORT = STAGE2_BASE_NUM_FEATURES + ["pred_effort"]

# Features removed because they duplicate another feature (checked on the train
# split: correlation and variance inflation factor, VIF). Kept here so the
# thesis can cite the exact reason for each removal.
REMOVED_FEATURES = {
    "priority_ordinal": "same information as the priority one-hot columns (priority entered twice)",
    "has_components": "r = 0.95 with num_components (VIF 10.5)",
    "has_labels": "r = 0.91 with num_labels (VIF 5.7)",
    "had_early_status_change": "exactly equal to (early_status_changes > 0); r = -0.90 with first_status_change_days_imputed",
}

TFIDF_PARAMS = dict(max_features=300, stop_words="english", ngram_range=(1, 2), sublinear_tf=True)


def add_full_text(df):
    df = df.copy()
    df["full_text"] = df["summary"].fillna("") + " " + df["description"].fillna("")
    return df


def build_preprocessor(num_features, use_text=False):
    transformers = [
        ("cat", OneHotEncoder(drop=CAT_BASELINES, handle_unknown="ignore"), CAT_FEATURES),
        ("num", StandardScaler(), num_features),
    ]
    if use_text:
        transformers.append(("text", TfidfVectorizer(**TFIDF_PARAMS), "full_text"))
    return ColumnTransformer(transformers=transformers)


# -----------------------------------------------------------------------------
# CANDIDATE MODELS (the same candidates are used everywhere)
# -----------------------------------------------------------------------------
def _lgbm_regressor():
    from lightgbm import LGBMRegressor
    return LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.05,
                         random_state=RANDOM_SEED, verbose=-1)


def _lgbm_classifier():
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=100, is_unbalance=True,
                          random_state=RANDOM_SEED, verbose=-1)


# name -> (uses TF-IDF text?, model factory)
EFFORT_CANDIDATES = {
    "Ridge (Tabular Only)": (False, lambda: Ridge(alpha=10.0, random_state=RANDOM_SEED)),
    "Ridge (Tabular + TF-IDF)": (True, lambda: Ridge(alpha=10.0, random_state=RANDOM_SEED)),
    "LightGBM (Tabular Only)": (False, _lgbm_regressor),
    "LightGBM (Tabular + TF-IDF)": (True, _lgbm_regressor),
}

RISK_CANDIDATES = {
    "Logistic Regression": lambda: LogisticRegression(class_weight="balanced", max_iter=1000,
                                                      random_state=RANDOM_SEED),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                                    random_state=RANDOM_SEED, n_jobs=-1),
    "LightGBM": _lgbm_classifier,
}
RISK_VARIANTS = {"No Effort": STAGE2_BASE_NUM_FEATURES, "+ Pred Effort": STAGE2_NUM_FEATURES_WITH_EFFORT}


def build_effort_pipeline(name):
    use_text, factory = EFFORT_CANDIDATES[name]
    return Pipeline([("preprocessor", build_preprocessor(STAGE1_NUM_FEATURES, use_text)),
                     ("model", factory())])


def build_risk_pipeline(family, with_effort=True):
    num = STAGE2_NUM_FEATURES_WITH_EFFORT if with_effort else STAGE2_BASE_NUM_FEATURES
    return Pipeline([("preprocessor", build_preprocessor(num)),
                     ("model", RISK_CANDIDATES[family]())])


def risk_name(family, variant):
    return f"{family} ({variant})"


# -----------------------------------------------------------------------------
# STAGE-1 -> STAGE-2 HAND-OFF
# -----------------------------------------------------------------------------
def predict_effort(pipe, df):
    """Effort models are trained on log1p(SP); convert back to Story Points."""
    return np.expm1(pipe.predict(df)).clip(min=0)


def oof_effort_predictions(df, effort_name, n_splits=5):
    """Out-of-fold predicted effort: no row is predicted by a model that saw it."""
    df = df.reset_index(drop=True)
    oof = np.zeros(len(df))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    for tr_idx, te_idx in kf.split(df):
        pipe = build_effort_pipeline(effort_name)
        pipe.fit(df.iloc[tr_idx], np.log1p(df.iloc[tr_idx]["story_points"].values))
        oof[te_idx] = predict_effort(pipe, df.iloc[te_idx])
    return oof


def risk_population(df):
    """Done issues still open after the observation window (leak-safe risk population)."""
    out = df[(df[RISK_LABEL].notna()) & (df["eligible_for_risk_model"] == 1)].copy().reset_index(drop=True)
    out[RISK_LABEL] = out[RISK_LABEL].astype(int)
    return out


# -----------------------------------------------------------------------------
# METRICS
# -----------------------------------------------------------------------------
def eval_effort_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    safe_y_true = np.where(y_true == 0, 1e-6, y_true)
    rel_errors = np.abs(y_true - y_pred) / safe_y_true
    return {
        "MAE": round(mean_absolute_error(y_true, y_pred), 4),
        "RMSE": round(root_mean_squared_error(y_true, y_pred), 4),
        "MdAE": round(median_absolute_error(y_true, y_pred), 4),
        "R2": round(r2_score(y_true, y_pred), 4),
        "PRED(25)%": round(np.mean(rel_errors <= 0.25) * 100.0, 2),
        "MdMRE%": round(np.median(rel_errors) * 100.0, 2),
    }


def eval_risk_metrics(y_true, y_pred, y_prob):
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = 0.5
    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except ValueError:
        pr_auc = float(np.mean(y_true))
    return {
        "ROC-AUC": round(roc_auc, 4), "PR-AUC": round(pr_auc, 4),
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "AtRisk-Precision": round(precision_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "AtRisk-Recall": round(recall_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "AtRisk-F1": round(f1_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "Macro-F1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "Brier": round(brier_score_loss(y_true, y_prob), 4),
    }


# -----------------------------------------------------------------------------
# MODEL SELECTION (decided on VALIDATION only, then frozen)
# -----------------------------------------------------------------------------
def save_selection(selection, path=SELECTION_FILE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(selection, f, indent=2)


def load_selection(path=SELECTION_FILE):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run src/train_advanced_models.py first - it selects the "
            "models on the validation split and writes this file."
        )
    with open(path) as f:
        return json.load(f)


# -----------------------------------------------------------------------------
# SHAP (works for both linear and tree models)
# -----------------------------------------------------------------------------
def to_dense(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


def shap_values_positive_class(model, X_background, X_explain):
    """
    SHAP values for the 'at risk' class, in log-odds units, shape (n_rows, n_features).
    - Logistic Regression -> LinearExplainer (exact for linear models)
    - Random Forest / LightGBM -> TreeExplainer
    Handles the different output shapes returned by SHAP for different models.
    """
    import shap

    X_background, X_explain = to_dense(X_background), to_dense(X_explain)
    if isinstance(model, LogisticRegression):
        values = shap.LinearExplainer(model, X_background).shap_values(X_explain)
    else:
        values = shap.TreeExplainer(model).shap_values(X_explain)
    if isinstance(values, list):          # older SHAP: one array per class
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:                   # newer SHAP for sklearn trees: (n, features, classes)
        values = values[:, :, 1]
    return values
