"""
predict.py - single-ticket inference for the effort & risk models.

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

import joblib
import numpy as np
import pandas as pd

from common import shap_values_positive_class

MODELS_DIR = "models"


def load_models():
    with open(f"{MODELS_DIR}/model_config.json") as f:
        config = json.load(f)
    stage1 = joblib.load(f"{MODELS_DIR}/stage1_effort_model.joblib")
    stage2 = joblib.load(f"{MODELS_DIR}/stage2_risk_model.joblib")
    return stage1, stage2, config


def _ticket_to_stage1_frame(ticket: dict) -> pd.DataFrame:
    """
    ticket: dict with the raw fields known at creation time, e.g.
        {
            "issuetype": "Story", "priority": "Minor",
            "num_components": 0, "num_labels": 0,
            "summary": "...", "description": "...",
            "reporter_prev_issue_count": 42,
        }
    The small derived fields are filled in automatically.
    """
    summary = ticket.get("summary", "") or ""
    description = ticket.get("description", "") or ""
    row = {
        "issuetype": ticket.get("issuetype", "Story"),
        "priority": ticket.get("priority", "Minor"),
        "num_components": ticket.get("num_components", 0),
        "num_labels": ticket.get("num_labels", 0),
        "summary_len": len(summary),
        "description_len": len(description),
        "reporter_prev_issue_count": ticket.get("reporter_prev_issue_count", 0),
        "has_description": int(len(description) > 0),
        "full_text": f"{summary} {description}",
    }
    return pd.DataFrame([row])


def predict_effort(ticket: dict, stage1) -> float:
    """Returns predicted Story Points. Usable the instant a ticket is created."""
    pred_log = stage1.predict(_ticket_to_stage1_frame(ticket))[0]
    return max(float(np.expm1(pred_log)), 0.0)


def _ticket_to_stage2_frame(ticket: dict, early_signals: dict, pred_effort: float, config: dict) -> pd.DataFrame:
    """
    early_signals: dict describing activity observed so far, e.g.
        {
            "early_status_changes": 1, "early_assignee_changes": 1,
            "early_priority_changes": 0, "early_comment_count": 2,
            "first_status_change_days": 1.4,   # None/omit if no status change yet
        }
    If a ticket has no activity yet, pass an empty dict - the same "no activity
    observed" default used in training is applied (first status change imputed
    to the full observation window).
    """
    row = _ticket_to_stage1_frame(ticket).iloc[0].to_dict()
    first_change_days = early_signals.get("first_status_change_days")
    row.update({
        "early_status_changes": early_signals.get("early_status_changes", 0),
        "early_assignee_changes": early_signals.get("early_assignee_changes", 0),
        "early_priority_changes": early_signals.get("early_priority_changes", 0),
        "early_comment_count": early_signals.get("early_comment_count", 0),
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
    pred_effort = predict_effort(ticket, stage1)
    frame = _ticket_to_stage2_frame(ticket, early_signals, pred_effort, config)
    prob = float(stage2.predict_proba(frame)[0, 1])
    return prob, int(prob >= 0.5), pred_effort


def explain_risk(ticket: dict, early_signals: dict, stage1, stage2, config: dict, top_n: int = 8):
    """
    Per-prediction SHAP explanation (log-odds units): the features that pushed
    THIS ticket's risk score up or down. Works for linear and tree models.
    """
    pred_effort = predict_effort(ticket, stage1)
    frame = _ticket_to_stage2_frame(ticket, early_signals, pred_effort, config)
    prep, model = stage2.named_steps["preprocessor"], stage2.named_steps["model"]
    background = joblib.load(f"{MODELS_DIR}/stage2_shap_background.joblib")
    shap_row = shap_values_positive_class(model, background, prep.transform(frame))[0]
    contributions = sorted(zip(prep.get_feature_names_out(), shap_row), key=lambda x: abs(x[1]), reverse=True)
    return contributions[:top_n]


def _example_to_inputs(row):
    ticket = {
        "issuetype": row["issuetype"], "priority": row["priority"],
        "num_components": row["num_components"], "num_labels": row["num_labels"],
        "reporter_prev_issue_count": row["reporter_prev_issue_count"],
        "summary": row["summary"] if pd.notna(row["summary"]) else "",
        "description": row["description"] if pd.notna(row["description"]) else "",
    }
    early_signals = {
        "early_status_changes": row["early_status_changes"],
        "early_assignee_changes": row["early_assignee_changes"],
        "early_priority_changes": row["early_priority_changes"],
        "early_comment_count": row["early_comment_count"],
        "first_status_change_days": (
            row["first_status_change_days_imputed"] if row["early_status_changes"] > 0 else None
        ),
    }
    return ticket, early_signals


def _run_sanity_check():
    """
    Runs the saved example tickets through the full inference path. These tickets
    were part of the training data, so this is a smoke test of the inference code,
    not an evaluation of accuracy (see evaluate_test_set.py for that).
    """
    stage1, stage2, config = load_models()
    examples = pd.read_csv(f"{MODELS_DIR}/example_tickets_for_sanity_check.csv")

    print("=" * 90)
    print(f"SANITY CHECK: predict.py vs known example tickets  "
          f"[Stage 1: {config['stage1_model']} | Stage 2: {config['stage2_model']}]")
    print("=" * 90)

    for _, row in examples.iterrows():
        ticket, early_signals = _example_to_inputs(row)
        prob, label, pred_effort = predict_risk(ticket, early_signals, stage1, stage2, config)
        print(f"\n{row['key']}  |  {ticket['summary'][:60]}")
        print(f"  Predicted Story Points: {pred_effort:.2f}   (actual: {row['story_points']})")
        print(f"  Predicted risk prob:    {prob:.3f}  -> label {label}   (actual: {int(row['risk_25pct'])})")

    first = examples.iloc[0]
    ticket, early_signals = _example_to_inputs(first)
    print("\n" + "=" * 90)
    print(f"SHAP explanation for {first['key']} (why this specific prediction, log-odds):")
    print("=" * 90)
    for feature, value in explain_risk(ticket, early_signals, stage1, stage2, config):
        direction = "pushes risk UP" if value > 0 else "pushes risk DOWN"
        print(f"  {feature:40s}  {value:+.4f}   ({direction})")


if __name__ == "__main__":
    _run_sanity_check()
