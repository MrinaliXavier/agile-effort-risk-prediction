"""
explain.py - per-ticket explanations for the two-stage pipeline (shared code).

For one ticket it answers, in plain words:
  1. How likely is it to miss its sprint?                   (Stage 2 probability)
  2. WHY - which facts push that risk up or down?            (SHAP on Stage 2, log-odds)
  3. How big does the model think the ticket is, compared
     with what the team committed?                          (Stage 1 estimate vs team SP)
  4. WHY does the model think it is that big?               (SHAP on Stage 1: fields + words)

SHAP values are additive:  model output = base value + sum of the feature contributions.
  - Stage 2 output is in LOG-ODDS of spilling over (positive = raises the risk).
  - Stage 1 output is log(1 + Story Points); a contribution c multiplies the estimate by about e^c.
Linear models (Ridge, Logistic Regression) use shap.LinearExplainer (exact for linear models);
tree models (Random Forest, LightGBM) use shap.TreeExplainer.
"""
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge

from common import shap_values_positive_class, to_dense

READABLE = {
    "num_components": "number of components",
    "num_labels": "number of labels",
    "summary_len": "summary length (characters)",
    "description_len": "description length (characters)",
    "reporter_prev_issue_count": "reporter's earlier tickets",
    "has_description": "has a description",
    "early_status_changes": "status changes before commitment",
    "early_assignee_changes": "assignee changes before commitment",
    "early_priority_changes": "priority changes before commitment",
    "early_comment_count": "comments before commitment",
    "first_status_change_days_imputed": "days until first status change",
    "backlog_days": "days in backlog before the sprint",
    "sprint_length_days": "sprint length (days)",
    "pred_effort": "model's effort estimate (SP)",
    "log_sp_team": "team's estimate at commitment (SP)",
    "log_sp_model": "model's effort estimate (SP)",
    "sp_missing_team": "team had not estimated it yet",
    "estimation_gap": "team estimate vs model estimate",
    "log_sp_final": "final Story Points (leakage check only)",
}
GROUPS = {"issuetype": "issue type", "priority": "priority"}

# Features that carry ONE fact between them are shown as one line (SHAP values add up, so the
# combined contribution is exact):
#  - log_sp_team + sp_missing_team: the team's estimate (log_sp_team is 0 when not estimated,
#    so the two columns split the "not estimated" effect between them)
#  - backlog_days + first_status_change_days_imputed: with no status change before the sprint,
#    the second is set to the first, so they move together
PAIRS = {
    "team": ({"num__log_sp_team", "num__sp_missing_team"}, "team's estimate at commitment"),
    "backlog": ({"num__backlog_days", "num__first_status_change_days_imputed"},
                "time in backlog before the sprint"),
}
GROUP_LABEL = {name: label for names, label in PAIRS.values() for name in names}


def readable(feature_name):
    """'cat__priority_Major' -> 'priority = Major', 'text__deploy' -> "word 'deploy'", 'num__x' -> label."""
    kind, _, name = feature_name.partition("__")
    if kind == "cat":
        col, _, value = name.partition("_")
        return f"{GROUPS.get(col, col)} = {value}"
    if kind == "text":
        return f"word '{name}'"
    return READABLE.get(name, name)


def display(feature_name, x):
    """Readable label that respects the ticket's own value: a one-hot column the ticket does NOT
    have reads 'issue type is not Bug' (its SHAP value is the effect of not being a Bug)."""
    kind, _, name = feature_name.partition("__")
    if kind == "cat" and x == 0:
        col, _, value = name.partition("_")
        return f"{GROUPS.get(col, col)} is not {value}"
    return readable(feature_name)


def group_of(feature_name):
    """Feature group for global importance: one-hot columns -> their source column, words -> 'description words'."""
    if feature_name in GROUP_LABEL:
        return GROUP_LABEL[feature_name]
    kind, _, name = feature_name.partition("__")
    if kind == "cat":
        return GROUPS.get(name.partition("_")[0], name)
    if kind == "text":
        return "words in summary/description"
    return READABLE.get(name, name)


def raw_value(feature_name, row):
    """The ticket's own value for a feature, in the units people know (not scaled)."""
    kind, _, name = feature_name.partition("__")
    if kind in ("cat", "text"):
        return None
    if name == "log_sp_team":
        v = row.get("sp_team_commit", row.get("sp_team_day5"))
        return "not estimated" if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), 1)
    if name == "log_sp_model":
        return round(float(row["pred_effort"]), 1)
    v = row.get(name)
    if v is None:
        return None
    try:
        return round(float(v), 1)
    except (TypeError, ValueError):
        return v


def shap_values_regression(model, X_background, X_explain):
    """SHAP for a Stage 1 regressor, shape (n_rows, n_features), in log(1+SP) units."""
    import shap

    X_background, X_explain = to_dense(X_background), to_dense(X_explain)
    if isinstance(model, (Ridge, LinearRegression)):
        try:
            masker = shap.maskers.Independent(X_background, max_samples=len(X_background))
        except AttributeError:
            masker = X_background
        values = shap.LinearExplainer(model, masker).shap_values(X_explain)
    else:
        values = shap.TreeExplainer(model).shap_values(X_explain)
    values = np.asarray(values)
    return values[0] if values.ndim == 3 else values


def units(stage2):
    """Stage-2 SHAP units: Random Forest explains the probability; LR and LightGBM the log-odds."""
    return "probability" if "RandomForest" in type(stage2.named_steps["model"]).__name__ else "log-odds"


def merge_pairs(contribs, row):
    """Combine each PAIRS group into one (label, shap, raw text) line; other features unchanged."""
    out, pooled = [], {}
    for name, v, x in contribs:
        key = next((k for k, (names, _) in PAIRS.items() if name in names), None)
        if key is None:
            val = raw_value(name, row)
            out.append((display(name, x), float(v), val))
        else:
            pooled[key] = pooled.get(key, 0.0) + float(v)
    if "team" in pooled:
        out.append((PAIRS["team"][1], pooled["team"], raw_value("num__log_sp_team", row)))
    if "backlog" in pooled:
        b = raw_value("num__backlog_days", row)
        f = raw_value("num__first_status_change_days_imputed", row)
        text = (f"{b} days" if b == f else f"{b} days (first status change after {f} days)") if b is not None else None
        out.append((PAIRS["backlog"][1], pooled["backlog"], text))
    return sorted(out, key=lambda x: abs(x[1]), reverse=True)


def stage2_contributions(stage2, background2, frame):
    """[(feature_name, shap_log_odds, transformed_value)] for the single row in `frame`, sorted by size."""
    prep, model = stage2.named_steps["preprocessor"], stage2.named_steps["model"]
    X = to_dense(prep.transform(frame))
    values = shap_values_positive_class(model, background2, X)[0]
    names = prep.get_feature_names_out()
    return sorted(zip(names, values, X[0]), key=lambda x: abs(x[1]), reverse=True)


def stage1_contributions(stage1, background1, frame):
    """[(feature_name, shap_log_sp, transformed_value)] for the single row; words only when they appear in the ticket."""
    prep, model = stage1.named_steps["preprocessor"], stage1.named_steps["model"]
    X = to_dense(prep.transform(frame))
    values = shap_values_regression(model, background1, X)[0]
    names = prep.get_feature_names_out()
    out = []
    for name, v, x in zip(names, values, X[0]):
        if name.startswith("text__") and x == 0:
            continue                       # a word the ticket does not contain
        out.append((name, float(v), x))
    return sorted(out, key=lambda x: abs(x[1]), reverse=True)


def explain_ticket(row_frame, stage1, stage2, background1, background2, project_spill_rate=None, top_n=6):
    """
    row_frame: a one-row DataFrame that already holds pred_effort and the estimate features
    (add_estimate_features). Returns (dict, text) - the dict for reports, the text for people.
    """
    row = row_frame.iloc[0].to_dict()
    prob = float(stage2.predict_proba(row_frame)[0, 1])
    risk = stage2_contributions(stage2, background2, row_frame)
    effort = stage1_contributions(stage1, background1, row_frame)

    team = row.get("sp_team_commit", row.get("sp_team_day5"))
    team_known = team is not None and not (isinstance(team, float) and np.isnan(team))
    model_sp = float(row["pred_effort"])

    lines = [f"Ticket {row.get('key', '(new)')}: {str(row.get('summary', ''))[:80]}",
             f"  Chance of missing its first sprint: {prob:.0%}"
             + (f"  (this project's usual rate: {project_spill_rate:.0%})" if project_spill_rate is not None else "")]
    if team_known:
        compare = (f" - {model_sp / float(team):.1f}x the team's estimate" if float(team) > 0 else "")
        lines.append(f"  Team committed {float(team):g} SP; the model estimates {model_sp:.1f} SP{compare}")
    else:
        lines.append(f"  The team had not estimated it at commitment; the model estimates {model_sp:.1f} SP")

    unit = units(stage2)
    risk_lines = merge_pairs(risk, row)
    if unit == "probability":
        lines.append("  What drives the risk (percentage points of probability; + raises it, - lowers it):")
        for label, v, val in risk_lines[:top_n]:
            lines.append(f"    {100 * v:+.1f} pts  {label}" + (f" = {val}" if val is not None else ""))
    else:
        lines.append("  What drives the risk (log-odds; + raises it, - lowers it):")
        for label, v, val in risk_lines[:top_n]:
            lines.append(f"    {v:+.3f}  {label}" + (f" = {val}" if val is not None else ""))

    lines.append("  Why the model estimates this size (each item multiplies the estimate by about e^value):")
    for name, v, x in effort[:top_n]:
        val = raw_value(name, row)
        lines.append(f"    {v:+.3f}  (x{np.exp(v):.2f})  {display(name, x)}" + (f" = {val}" if val is not None else ""))

    result = {
        "key": row.get("key"), "probability": round(prob, 4),
        "team_sp_at_commitment": float(team) if team_known else None, "model_sp": round(model_sp, 2),
        "risk_units": unit,
        "risk_drivers": [{"feature": label, "shap": round(float(v), 4), "value": val} for label, v, val in risk_lines[:top_n]],
        "effort_drivers": [{"feature": display(n, x), "shap_log_sp": round(float(v), 4)} for n, v, x in effort[:top_n]],
    }
    return result, "\n".join(lines)


def global_importance(stage2, background2, frame):
    """Mean |group SHAP| over many rows (e.g. a test set), normalised to sum to 1. A group's SHAP for a
    row is the sum of its features' SHAP values (one-hot columns of one field, a PAIRS group)."""
    import pandas as pd

    prep, model = stage2.named_steps["preprocessor"], stage2.named_steps["model"]
    values = shap_values_positive_class(model, background2, prep.transform(frame))
    # add each group's parts per ticket FIRST, then take |.| - parts that cancel do not inflate a group
    per_row = pd.DataFrame(values, columns=[group_of(n) for n in prep.get_feature_names_out()])
    per_row = per_row.T.groupby(level=0).sum().T
    out = per_row.abs().mean(axis=0).rename("mean_abs_shap").rename_axis("group").reset_index()
    out["share"] = out["mean_abs_shap"] / out["mean_abs_shap"].sum()
    return out.sort_values("share", ascending=False).reset_index(drop=True)
