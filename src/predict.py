"""
predict.py — single-ticket inference for the effort & risk models.

IMPORTANT DESIGN NOTE (read before using):
    Effort can be predicted the instant a ticket is created.
    Risk is a 5-day early-warning check: it needs some initial activity
    (status changes, assignee changes, comments observed so far) to be
    meaningful, by design (see model_config.json -> early_window_days).
    Calling predict_risk() with no early-activity signals at all is valid
    but will reflect a ticket that has shown no activity yet, not a magic
    answer for a ticket that was literally just filed seconds ago.
"""

import json
import numpy as np
import pandas as pd
import joblib
import shap

MODELS_DIR = "models"


def load_models():
    with open(f"{MODELS_DIR}/model_config.json") as f:
        config = json.load(f)
    stage1 = joblib.load(f"{MODELS_DIR}/stage1_effort_model.joblib")
    stage2 = joblib.load(f"{MODELS_DIR}/stage2_risk_model.joblib")
    return stage1, stage2, config


def _ticket_to_stage1_frame(ticket: dict, config: dict) -> pd.DataFrame:
    """
    ticket: dict with the raw fields known at creation time, e.g.
        {
            "issuetype": "Story", "priority": "Minor",
            "num_components": 0, "num_labels": 0,
            "summary": "...", "description": "...",
            "reporter_prev_issue_count": 42,
        }
    Fills in the small derived fields automatically so the caller doesn't
    have to compute them by hand.
    """
    summary = ticket.get("summary", "") or ""
    description = ticket.get("description", "") or ""

    priority_map = {"Trivial": 1, "Minor": 2, "Major": 3, "Critical": 4, "Blocker": 5}
    priority = ticket.get("priority", "Minor")

    row = {
        "issuetype": ticket.get("issuetype", "Story"),
        "priority": priority,
        "num_components": ticket.get("num_components", 0),
        "num_labels": ticket.get("num_labels", 0),
        "summary_len": len(summary),
        "description_len": len(description),
        "reporter_prev_issue_count": ticket.get("reporter_prev_issue_count", 0),
        "priority_ordinal": priority_map.get(priority, 2),
        "has_description": int(len(description) > 0),
        "has_components": int(ticket.get("num_components", 0) > 0),
        "has_labels": int(ticket.get("num_labels", 0) > 0),
        "full_text": f"{summary} {description}",
    }
    return pd.DataFrame([row])


def predict_effort(ticket: dict, stage1, config: dict) -> float:
    """Returns predicted Story Points. Usable the instant a ticket is created."""
    frame = _ticket_to_stage1_frame(ticket, config)
    pred_log = stage1.predict(frame)[0]
    pred = float(np.expm1(pred_log))
    return max(pred, 0.0)


def _ticket_to_stage2_frame(ticket: dict, early_signals: dict, pred_effort: float, config: dict) -> pd.DataFrame:
    """
    early_signals: dict describing activity observed so far, e.g.
        {
            "early_status_changes": 1, "early_assignee_changes": 1,
            "early_priority_changes": 0, "early_comment_count": 2,
            "first_status_change_days": 1.4,   # None/omit if no status change yet
        }
    If a ticket genuinely has no activity yet, pass an empty dict - the
    function fills in the same "no activity observed" defaults used in
    training (had_early_status_change=0, first_status_change_days imputed
    to the full observation window).
    """
    stage1_frame = _ticket_to_stage1_frame(ticket, config)
    row = stage1_frame.iloc[0].to_dict()

    first_change_days = early_signals.get("first_status_change_days")
    row.update({
        "early_status_changes": early_signals.get("early_status_changes", 0),
        "early_assignee_changes": early_signals.get("early_assignee_changes", 0),
        "early_priority_changes": early_signals.get("early_priority_changes", 0),
        "early_comment_count": early_signals.get("early_comment_count", 0),
        "had_early_status_change": int(first_change_days is not None),
        "first_status_change_days_imputed": (
            float(first_change_days) if first_change_days is not None else float(config["early_window_days"])
        ),
        "pred_effort": pred_effort,
    })
    return pd.DataFrame([row])


def predict_risk(ticket: dict, early_signals: dict, stage1, stage2, config: dict):
    """
    Returns (risk_probability, risk_label, pred_effort_used).
    This is the 5-day early-warning check - see module docstring.
    """
    pred_effort = predict_effort(ticket, stage1, config)
    frame = _ticket_to_stage2_frame(ticket, early_signals, pred_effort, config)
    prob = float(stage2.predict_proba(frame)[0, 1])
    label = int(prob >= 0.5)
    return prob, label, pred_effort


def explain_risk(ticket: dict, early_signals: dict, stage1, stage2, config: dict, top_n: int = 8):
    """
    Returns a per-prediction SHAP explanation: the specific features that
    pushed THIS ticket's risk score up or down, not just global importance.
    """
    pred_effort = predict_effort(ticket, stage1, config)
    frame = _ticket_to_stage2_frame(ticket, early_signals, pred_effort, config)

    prep = stage2.named_steps["preprocessor"]
    model = stage2.named_steps["model"]

    cat_encoder = prep.named_transformers_["cat"]
    cat_names = list(cat_encoder.get_feature_names_out(config["cat_features"]))
    all_feature_names = cat_names + config["stage2_num_features_with_effort"]

    X_trans = prep.transform(frame)
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_trans)
    shap_row = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]

    contributions = sorted(
        zip(all_feature_names, shap_row), key=lambda x: abs(x[1]), reverse=True
    )
    return contributions[:top_n]


def _run_sanity_check():
    """
    Runs every saved example ticket through the full inference path and
    compares predictions to the known outcome - confirms the saved models
    behave sensibly outside of the training script itself.
    """
    stage1, stage2, config = load_models()
    examples = pd.read_csv(f"{MODELS_DIR}/example_tickets_for_sanity_check.csv")

    print("=" * 90)
    print("SANITY CHECK: predict.py vs known example tickets")
    print("=" * 90)

    for _, row in examples.iterrows():
        ticket = {
            "issuetype": row["issuetype"],
            "priority": row["priority"],
            "num_components": row["num_components"],
            "num_labels": row["num_labels"],
            "reporter_prev_issue_count": row["reporter_prev_issue_count"],
            "summary": row["summary"],
            "description": row["description"] if pd.notna(row["description"]) else "",
        }
        early_signals = {
            "early_status_changes": row["early_status_changes"],
            "early_assignee_changes": row["early_assignee_changes"],
            "early_priority_changes": row["early_priority_changes"],
            "early_comment_count": row["early_comment_count"],
            "first_status_change_days": (
                row["first_status_change_days_imputed"] if row["had_early_status_change"] == 1 else None
            ),
        }

        prob, label, pred_effort = predict_risk(ticket, early_signals, stage1, stage2, config)

        print(f"\n{row['key']}  |  {ticket['summary'][:60]}")
        print(f"  Predicted Story Points: {pred_effort:.2f}   (actual: {row['story_points']})")
        print(f"  Predicted risk prob:    {prob:.3f}  -> label {label}   (actual: {int(row['risk_25pct'])})")

    # Show one full SHAP explanation as a demo of the explainability path
    first = examples.iloc[0]
    ticket = {
        "issuetype": first["issuetype"], "priority": first["priority"],
        "num_components": first["num_components"], "num_labels": first["num_labels"],
        "reporter_prev_issue_count": first["reporter_prev_issue_count"],
        "summary": first["summary"], "description": first["description"] if pd.notna(first["description"]) else "",
    }
    early_signals = {
        "early_status_changes": first["early_status_changes"],
        "early_assignee_changes": first["early_assignee_changes"],
        "early_priority_changes": first["early_priority_changes"],
        "early_comment_count": first["early_comment_count"],
        "first_status_change_days": (
            first["first_status_change_days_imputed"] if first["had_early_status_change"] == 1 else None
        ),
    }
    print("\n" + "=" * 90)
    print(f"SHAP explanation for {first['key']} (why this specific prediction):")
    print("=" * 90)
    for feature, value in explain_risk(ticket, early_signals, stage1, stage2, config):
        direction = "pushes risk UP" if value > 0 else "pushes risk DOWN"
        print(f"  {feature:35s}  {value:+.4f}   ({direction})")


if __name__ == "__main__":
    _run_sanity_check()