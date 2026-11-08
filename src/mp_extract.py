"""
mp_extract.py - build the MULTI-PROJECT modelling dataset in one pass.

For every project in projects_config.py this does what build_labels_and_splits.py,
build_features.py and prepare_modeling_dataset.py did for XD:

  1. Pull every ticket with an estimated Story Points value (+ changelog, comments, text)
  2. Drop Story Points outliers (> SP_MAX) and split chronologically 70/15/15 (per project)
  3. Keep only genuinely COMPLETED work for the delay label (resolution date, done-like
     status, resolution not Won't Fix / Duplicate / ...)
  4. Delay label per project:  log(1 + cycle_time) = a + b * SP   (fitted on that project's
     training tickets only); at-risk = top 25% of training residuals
  5. Creation-time + first-5-days features (same definitions as XD)
  6. NEW - the team's Story Points AS THEY STOOD ON DAY 5, rebuilt from the changelog,
     so the estimation-gap feature never uses a later re-estimate (no leakage)

Run from the project root (MongoDB must be running):
    python src/mp_extract.py                 # all projects
    python src/mp_extract.py XD KEYCLOAK     # only some projects (quick test)

Output:
    data/processed/mp_modeling_dataset.csv   one row per ticket, with a 'project' column
    reports/mp_extraction_summary.csv        per-project diagnostics
    reports/mp_excluded_resolutions.csv      which resolutions were excluded, per project
"""
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from pymongo import MongoClient
from sklearn.linear_model import LinearRegression

from projects_config import (DONE_STATUSES, EARLY_WINDOW_DAYS, SP_FIELDS, SP_MAX, active_projects,
                             resolution_is_excluded)

client = MongoClient("mongodb://localhost:27017/")
db = client["JiraReposAnon"]
COLLECTIONS = {name.lower(): name for name in db.list_collection_names()}


# -----------------------------------------------------------------------------
# small helpers
# -----------------------------------------------------------------------------
def to_utc(value):
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    return None if pd.isna(ts) else ts


def to_number(value):
    if value is None:
        return None
    try:
        text = str(value).strip()
        return float(text) if text not in ("", "None", "null") else None
    except ValueError:
        return None


def first_number(fields, field_ids):
    for fid in field_ids:
        v = to_number(fields.get(fid))
        if v is not None:
            return v
    return None


def is_story_point_item(item, field_ids):
    if item.get("fieldId") in field_ids:
        return True
    name = str(item.get("field") or "").lower()
    return "story point" in name and "actual" not in name and "original" not in name


def parse_histories(histories):
    """Parse each changelog entry's timestamp once: [(timestamp, history), ...] sorted by time."""
    parsed = [(to_utc(h.get("created")), h) for h in histories or []]
    return sorted([(t, h) for t, h in parsed if t is not None], key=lambda x: x[0])


def story_points_as_of(final_sp, parsed_histories, cutoff, field_ids):
    """
    Rebuild the Story Points value at `cutoff` by walking the changelog BACKWARDS
    from the final value and undoing every Story Points change made after the cutoff.

    Returns (sp_at_cutoff or None if not yet estimated, total number of SP changes,
             whether SP was changed after the cutoff).
    """
    changes = [(t, item) for t, h in parsed_histories
               for item in (h.get("items") or []) if is_story_point_item(item, field_ids)]
    value, changed_after = final_sp, False
    for t, item in reversed(changes):
        if t <= cutoff:
            break
        before = item.get("fromString")
        value = to_number(before if before is not None else item.get("from"))
        changed_after = True
    return value, len(changes), changed_after


def early_window_features(created, parsed_histories, comments):
    """Same definitions as build_features.py (XD)."""
    cutoff = created + pd.Timedelta(days=EARLY_WINDOW_DAYS)
    status = assignee = priority = 0
    first_status = None
    for t, h in parsed_histories:
        if t > cutoff:
            break
        for item in h.get("items") or []:
            field = item.get("field")
            if field == "status":
                status += 1
                if first_status is None:
                    first_status = (t - created).total_seconds() / 86400
            elif field == "assignee":
                assignee += 1
            elif field == "priority":
                priority += 1
    n_comments = 0
    for c in comments or []:
        t = to_utc(c.get("created"))
        if t is not None and t <= cutoff:
            n_comments += 1
    return {
        "early_status_changes": status,
        "early_assignee_changes": assignee,
        "early_priority_changes": priority,
        "early_comment_count": n_comments,
        "first_status_change_days": first_status,
    }


def chronological_split(df):
    df = df.sort_values("created").reset_index(drop=True)
    n = len(df)
    train_cut = df.loc[int(n * 0.70), "created"]
    val_cut = df.loc[int(n * 0.85), "created"]
    df["split"] = pd.cut(
        df["created"],
        bins=[df["created"].min() - pd.Timedelta(days=1), train_cut, val_cut,
              df["created"].max() + pd.Timedelta(days=1)],
        labels=["train", "val", "test"],
    ).astype(str)
    return df


# -----------------------------------------------------------------------------
# one project
# -----------------------------------------------------------------------------
def extract_project(project):
    repo, key = project["repo"], project["key"]
    coll_name = COLLECTIONS.get(repo.lower())
    if coll_name is None:
        raise ValueError(f"collection for repository '{repo}' not found")
    coll = db[coll_name]
    sp_fields = SP_FIELDS[repo]

    projection = {
        "key": 1, "changelog.histories": 1,
        "fields.created": 1, "fields.resolutiondate": 1, "fields.status.name": 1, "fields.resolution.name": 1,
        "fields.issuetype.name": 1, "fields.priority.name": 1, "fields.components": 1, "fields.labels": 1,
        "fields.summary": 1, "fields.description": 1, "fields.reporter": 1,
        "fields.comments": 1, "fields.comment": 1,
    }
    for fid in sp_fields:
        projection[f"fields.{fid}"] = 1
    query = {"fields.project.key": key, "$or": [{f"fields.{fid}": {"$ne": None}} for fid in sp_fields]}

    rows = []
    for doc in coll.find(query, projection):
        f = doc.get("fields", {}) or {}
        sp = first_number(f, sp_fields)
        created = to_utc(f.get("created"))
        if sp is None or created is None:
            continue
        histories = parse_histories((doc.get("changelog") or {}).get("histories", []))
        comments = f.get("comments")
        if comments is None:
            comments = (f.get("comment") or {}).get("comments", [])
        reporter = f.get("reporter") or {}
        resolution = (f.get("resolution") or {}).get("name")
        summary, description = f.get("summary") or "", f.get("description") or ""

        sp_day5, n_sp_changes, sp_changed_after = story_points_as_of(
            sp, histories, created + pd.Timedelta(days=EARLY_WINDOW_DAYS), sp_fields)

        row = {
            "project": key, "repository": repo, "key": doc.get("key"),
            "created": created, "resolutiondate": to_utc(f.get("resolutiondate")),
            "status": (f.get("status") or {}).get("name"), "resolution": resolution,
            "story_points": sp,
            "sp_team_day5": sp_day5, "sp_history_changes": n_sp_changes,
            "sp_changed_after_window": int(sp_changed_after),
            "issuetype": (f.get("issuetype") or {}).get("name") or "Unknown",
            "priority": (f.get("priority") or {}).get("name") or "None",
            "num_components": len(f.get("components") or []),
            "num_labels": len(f.get("labels") or []),
            "summary": summary, "description": description,
            "summary_len": len(summary), "description_len": len(description),
            "reporter": reporter.get("key") or reporter.get("name") or reporter.get("accountId"),
        }
        row.update(early_window_features(created, histories, comments))
        rows.append(row)

    df = pd.DataFrame(rows)
    diag = {"project": key, "repository": repo, "docs_project": project["docs"],
            "tickets_with_sp": len(df)}

    # --- outliers ---
    keep = (df["story_points"] >= 0) & (df["story_points"] <= SP_MAX)
    diag["dropped_sp_outliers"] = int((~keep).sum())
    df = df[keep].copy()

    # --- chronological split (per project) ---
    df = chronological_split(df)

    # --- completed work only, for the delay label ---
    excluded_res = df["resolution"].apply(resolution_is_excluded)
    done_status = df["status"].fillna("").str.lower().isin(DONE_STATUSES)
    completed = df["resolutiondate"].notna() & done_status & ~excluded_res
    df["cycle_time_days"] = np.where(
        completed, (df["resolutiondate"] - df["created"]).dt.total_seconds() / 86400, np.nan)
    df.loc[df["cycle_time_days"] <= 0, "cycle_time_days"] = np.nan
    diag["completed"] = int(df["cycle_time_days"].notna().sum())
    diag["resolved_but_excluded_resolution"] = int((df["resolutiondate"].notna() & excluded_res).sum())
    diag["resolved_but_not_done_status"] = int((df["resolutiondate"].notna() & ~done_status & ~excluded_res).sum())
    diag["unresolved"] = int(df["resolutiondate"].isna().sum())
    excluded_counts = Counter(df.loc[df["resolutiondate"].notna() & excluded_res, "resolution"])

    # --- delay label: per-project log-linear fit on TRAIN completed tickets only ---
    done = df[df["cycle_time_days"].notna()]
    train_done = done[done["split"] == "train"]
    reg = LinearRegression().fit(train_done[["story_points"]], np.log1p(train_done["cycle_time_days"]))
    diag["label_intercept"] = round(float(reg.intercept_), 4)
    diag["label_slope_per_sp"] = round(float(reg.coef_[0]), 4)
    df["residual"] = np.nan
    df.loc[done.index, "residual"] = (np.log1p(done["cycle_time_days"]) -
                                      reg.predict(done[["story_points"]]))
    train_resid = df.loc[(df["split"] == "train") & df["residual"].notna(), "residual"]
    for pct in (20, 25, 30):
        cutoff = train_resid.quantile(1 - pct / 100)
        df[f"risk_{pct}pct"] = np.where(df["residual"].notna(), (df["residual"] > cutoff).astype(float), np.nan)

    # --- reporter history: number of this reporter's EARLIER tickets in the project ---
    # (strictly earlier tickets; a missing reporter counts as no history)
    df["reporter_prev_issue_count"] = (
        df.groupby(df["reporter"].fillna("__none__"))["created"].rank(method="min").astype(int) - 1)
    df.loc[df["reporter"].isna(), "reporter_prev_issue_count"] = 0

    # --- same derived features as XD ---
    df["first_status_change_days_imputed"] = df["first_status_change_days"].fillna(EARLY_WINDOW_DAYS)
    df["has_description"] = (df["description_len"] > 0).astype(int)
    df["eligible_for_risk_model"] = (df["cycle_time_days"].notna() &
                                     (df["cycle_time_days"] > EARLY_WINDOW_DAYS)).astype(int)

    # --- diagnostics ---
    elig = df[df["eligible_for_risk_model"] == 1]
    for split in ("train", "val", "test"):
        s = elig[elig["split"] == split]
        diag[f"eligible_{split}"] = len(s)
        diag[f"at_risk_rate_{split}"] = round(float(s["risk_25pct"].mean()), 3) if len(s) else None
    diag["sp_known_at_day5_pct"] = round(100 * df["sp_team_day5"].notna().mean(), 1)
    diag["sp_changed_after_day5_pct"] = round(100 * df["sp_changed_after_window"].mean(), 1)
    diag["sp_has_history_pct"] = round(100 * (df["sp_history_changes"] > 0).mean(), 1)
    return df, diag, excluded_counts


def main():
    only = {k.upper() for k in sys.argv[1:]}
    projects = [p for p in active_projects() if not only or p["key"] in only]

    frames, diags, excluded_rows = [], [], []
    for p in projects:
        print(f"\n=== {p['repo']} / {p['key']} ===")
        try:
            df, diag, excluded = extract_project(p)
        except Exception as exc:
            print(f"  !! failed: {exc}")
            continue
        frames.append(df)
        diags.append(diag)
        excluded_rows += [{"project": p["key"], "resolution": r, "count": c} for r, c in excluded.most_common()]
        print(f"  tickets {diag['tickets_with_sp']} | outliers dropped {diag['dropped_sp_outliers']} | "
              f"completed {diag['completed']} | excluded resolutions {diag['resolved_but_excluded_resolution']} | "
              f"unresolved {diag['unresolved']}")
        print(f"  label: log(1+cycle) = {diag['label_intercept']} + {diag['label_slope_per_sp']} * SP")
        print(f"  eligible train/val/test: {diag['eligible_train']}/{diag['eligible_val']}/{diag['eligible_test']} | "
              f"at-risk rate {diag['at_risk_rate_train']}/{diag['at_risk_rate_val']}/{diag['at_risk_rate_test']}")
        print(f"  SP known at day 5: {diag['sp_known_at_day5_pct']}% | changed after day 5: "
              f"{diag['sp_changed_after_day5_pct']}% | has SP history: {diag['sp_has_history_pct']}%")

    if not frames:
        print("Nothing extracted.")
        return
    out = pd.concat(frames, ignore_index=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    out.drop(columns=["reporter"]).to_csv("data/processed/mp_modeling_dataset.csv", index=False)
    pd.DataFrame(diags).to_csv("reports/mp_extraction_summary.csv", index=False)
    pd.DataFrame(excluded_rows).to_csv("reports/mp_excluded_resolutions.csv", index=False)

    print("\n" + "=" * 100)
    print(pd.DataFrame(diags)[["project", "tickets_with_sp", "completed", "eligible_train", "eligible_val",
                                "eligible_test", "sp_known_at_day5_pct", "sp_changed_after_day5_pct"]].to_string(index=False))
    print(f"\nSaved data/processed/mp_modeling_dataset.csv ({len(out)} tickets, {len(frames)} projects)")
    print("Saved reports/mp_extraction_summary.csv and reports/mp_excluded_resolutions.csv")


if __name__ == "__main__":
    main()
