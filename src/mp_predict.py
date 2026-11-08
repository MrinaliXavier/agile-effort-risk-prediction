"""
mp_predict.py - predict and EXPLAIN sprint-spillover risk for one ticket at sprint planning.

Use from Python:
    from mp_predict import load_project, predict_ticket
    models = load_project("XD")
    result, text = predict_ticket(models, {
        "summary": "Add retry logic to the batch job", "description": "...",
        "issuetype": "Story", "priority": "Major",
        "num_components": 1, "num_labels": 0, "reporter_prev_issue_count": 40,
        "sp_team_commit": 3,            # the team's Story Points at commitment (None if not estimated)
        "early_status_changes": 1, "early_assignee_changes": 1,      # activity BEFORE the sprint
        "early_priority_changes": 0, "early_comment_count": 2,
        "first_status_change_days": 2.0,  # None if the status has not changed yet
        "backlog_days": 6,              # days from creation to entering the sprint
        "sprint_length_days": 14,
    })
    print(text)

Smoke test from the command line (uses example tickets that were in the training data,
so it checks the code path, not accuracy - accuracy is in mp_train_evaluate.py):
    python src/mp_predict.py XD
"""
import json
import sys

import joblib
import numpy as np
import pandas as pd

from explain import explain_ticket
from common import add_estimate_features


def load_project(project, label_name="spillover"):
    folder = f"models/{label_name}/{project}"
    with open(f"{folder}/config.json") as f:
        config = json.load(f)
    return {"config": config,
            "stage1": joblib.load(f"{folder}/stage1.joblib"), "stage2": joblib.load(f"{folder}/stage2.joblib"),
            "bg1": joblib.load(f"{folder}/background1.joblib"), "bg2": joblib.load(f"{folder}/background2.joblib")}


def ticket_frame(ticket):
    """Raw ticket fields -> the one-row frame both pipelines expect."""
    summary, description = ticket.get("summary") or "", ticket.get("description") or ""
    first = ticket.get("first_status_change_days")
    backlog = float(ticket.get("backlog_days", 0))
    team = ticket.get("sp_team_commit")
    row = {
        "key": ticket.get("key", "(new)"), "summary": summary, "description": description,
        "full_text": f"{summary} {description}",
        "issuetype": str(ticket.get("issuetype", "Story")), "priority": str(ticket.get("priority", "None")),
        "num_components": ticket.get("num_components", 0), "num_labels": ticket.get("num_labels", 0),
        "summary_len": len(summary), "description_len": len(description), "has_description": int(len(description) > 0),
        "reporter_prev_issue_count": ticket.get("reporter_prev_issue_count", 0),
        "early_status_changes": ticket.get("early_status_changes", 0),
        "early_assignee_changes": ticket.get("early_assignee_changes", 0),
        "early_priority_changes": ticket.get("early_priority_changes", 0),
        "early_comment_count": ticket.get("early_comment_count", 0),
        "first_status_change_days_imputed": float(first) if first is not None else backlog,
        "backlog_days": backlog, "sprint_length_days": float(ticket.get("sprint_length_days", 14)),
        "sp_team_commit": np.nan if team is None else float(team),
        "story_points": np.nan,     # unknown at prediction time; used only by the leakage check
    }
    return pd.DataFrame([row])


def predict_ticket(models, ticket):
    frame = ticket_frame(ticket)
    frame["pred_effort"] = max(float(np.expm1(models["stage1"].predict(frame)[0])), 0.0)
    frame = add_estimate_features(frame, models["config"]["team_col"])
    return explain_ticket(frame, models["stage1"], models["stage2"], models["bg1"], models["bg2"],
                          project_spill_rate=models["config"]["historical_delay_rate"])


def _smoke_test(project):
    models = load_project(project)
    examples = pd.read_csv(f"models/spillover/{project}/example_tickets.csv")
    print(f"Smoke test - {project}: Stage 1 {models['config']['effort_model']} | "
          f"Stage 2 {models['config']['risk_family']} (these tickets were in the training data)\n")
    for _, r in examples.iterrows():
        t = r.to_dict()
        t["sp_team_commit"] = None if pd.isna(t.get("sp_team_commit")) else t["sp_team_commit"]
        t["first_status_change_days"] = t.get("first_status_change_days_imputed")
        _, text = predict_ticket(models, t)
        print(text)
        if "spilled" in t:
            print(f"  ACTUAL: {'missed its sprint' if t['spilled'] == 1 else 'finished in its sprint'}\n")


if __name__ == "__main__":
    _smoke_test(sys.argv[1].upper() if len(sys.argv) > 1 else "XD")
