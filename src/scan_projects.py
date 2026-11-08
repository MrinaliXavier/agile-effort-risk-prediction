"""
scan_projects.py - which projects in the Public Jira Dataset are usable for
effort + delay-risk modelling?

For every repository (MongoDB collection) and every project inside it, counts:
  - tickets with an estimated Story Points value
  - of those, how many were resolved (have a resolution date)
  - of those, how many have a "done-like" final status
  - tickets that still have NO resolution (unfinished = censored)
  - tickets that also have an "Actual" Story Points value, and how many of
    those differ from the estimate (= real estimation error)

Selection rule (decided BEFORE looking at any model results, to avoid
cherry-picking): a project qualifies if it has at least MIN_RESOLVED_WITH_SP
resolved tickets with an estimated Story Points value.

Run from the project root:   python src/scan_projects.py
Output: printed table + reports/project_scan.csv
"""
import json
import os

import pandas as pd
from pymongo import MongoClient

FIELD_INFO_PATH = os.path.join("data", "raw", "ThePublicJiraDataset", "0. DataDefinition",
                               "jira_field_information.json")
MIN_RESOLVED_WITH_SP = 1000      # selection rule, fixed in advance
MIN_PROJECT_SP_TICKETS = 100     # projects smaller than this are not printed
DONE_STATUSES = ["Done", "Closed", "Resolved", "Fixed", "Complete", "Completed"]

client = MongoClient("mongodb://localhost:27017/")
db = client["JiraReposAnon"]


def story_point_fields(field_info):
    """Split each repository's Story-Points-like fields into 'estimate' and 'actual'."""
    out = {}
    for repo, fields in field_info.items():
        est, act = [], []
        for f in fields:
            name = f["name"].lower()
            if "story point" not in name:
                continue
            if "actual" in name:
                act.append(f["id"])
            elif "original" in name:
                continue      # "original story points" is an older copy of the estimate; skip
            else:
                est.append(f["id"])
        if est:
            out[repo] = {"estimate": est, "actual": act}
    return out


def first_numeric(field_ids):
    """Mongo expression: the first of these fields that holds a number (else null)."""
    expr = None
    for fid in reversed(field_ids):
        conv = {"$convert": {"input": f"$fields.{fid}", "to": "double", "onError": None, "onNull": None}}
        expr = conv if expr is None else {"$ifNull": [conv, expr]}
    return expr


def is_set(expr):
    return {"$cond": [{"$ne": [{"$ifNull": [expr, None]}, None]}, 1, 0]}


def scan_repo(repo, sp):
    coll = db[repo]
    est_expr = first_numeric(sp["estimate"])
    act_expr = first_numeric(sp["actual"]) if sp["actual"] else None

    any_sp = [{f"fields.{fid}": {"$ne": None}} for fid in sp["estimate"]]
    resolved = is_set("$fields.resolutiondate")
    done = {"$cond": [{"$in": ["$fields.status.name", DONE_STATUSES]}, 1, 0]}

    project_stage = {
        "est": est_expr,
        "resolved": resolved,
        "done": done,
        "created": "$fields.created",
        "key": "$fields.project.key",
    }
    if act_expr is not None:
        project_stage["act"] = act_expr

    group_stage = {
        "_id": "$key",
        "sp_tickets": {"$sum": {"$cond": [{"$ne": ["$est", None]}, 1, 0]}},
        "sp_resolved": {"$sum": {"$cond": [{"$and": [{"$ne": ["$est", None]}, {"$eq": ["$resolved", 1]}]}, 1, 0]}},
        "sp_resolved_done": {"$sum": {"$cond": [{"$and": [{"$ne": ["$est", None]}, {"$eq": ["$resolved", 1]},
                                                           {"$eq": ["$done", 1]}]}, 1, 0]}},
        "sp_unresolved": {"$sum": {"$cond": [{"$and": [{"$ne": ["$est", None]}, {"$eq": ["$resolved", 0]}]}, 1, 0]}},
        "sp_mean": {"$avg": "$est"},
        "sp_max": {"$max": "$est"},
        "first_created": {"$min": "$created"},
        "last_created": {"$max": "$created"},
    }
    if act_expr is not None:
        both = {"$and": [{"$ne": ["$est", None]}, {"$ne": ["$act", None]}]}
        group_stage["with_actual_sp"] = {"$sum": {"$cond": [both, 1, 0]}}
        group_stage["actual_differs"] = {"$sum": {"$cond": [{"$and": [both, {"$ne": ["$est", "$act"]}]}, 1, 0]}}

    pipeline = [
        {"$match": {"$or": any_sp}},
        {"$project": project_stage},
        {"$group": group_stage},
    ]
    rows = []
    for r in coll.aggregate(pipeline, allowDiskUse=True):
        r["repository"] = repo
        r["project"] = r.pop("_id")
        rows.append(r)
    return rows


def main():
    with open(FIELD_INFO_PATH, encoding="utf-8") as f:
        sp_fields = story_point_fields(json.load(f))

    # field file and MongoDB spell some names differently (e.g. SecondLife vs Secondlife)
    existing = {name.lower(): name for name in db.list_collection_names()}
    all_rows = []
    for field_repo, sp in sp_fields.items():
        repo = existing.get(field_repo.lower())
        if repo is None:
            print(f"[skip] {field_repo}: no matching collection in JiraReposAnon")
            continue
        print(f"Scanning {repo} (estimate fields: {sp['estimate']}, actual fields: {sp['actual'] or 'none'}) ...")
        try:
            all_rows.extend(scan_repo(repo, sp))
        except Exception as exc:   # keep going if one repository fails
            print(f"  !! {repo} failed: {exc}")

    df = pd.DataFrame(all_rows)
    for col in ["with_actual_sp", "actual_differs"]:
        if col not in df:
            df[col] = 0
    df[["with_actual_sp", "actual_differs"]] = df[["with_actual_sp", "actual_differs"]].fillna(0).astype(int)
    df["unresolved_pct"] = (100 * df["sp_unresolved"] / df["sp_tickets"]).round(1)
    df["sp_mean"] = df["sp_mean"].round(2)
    df["first_created"] = df["first_created"].astype(str).str[:10]
    df["last_created"] = df["last_created"].astype(str).str[:10]
    df["qualifies"] = df["sp_resolved"] >= MIN_RESOLVED_WITH_SP
    df = df.sort_values("sp_resolved", ascending=False)

    cols = ["repository", "project", "sp_tickets", "sp_resolved", "sp_resolved_done", "sp_unresolved",
            "unresolved_pct", "with_actual_sp", "actual_differs", "sp_mean", "sp_max",
            "first_created", "last_created", "qualifies"]
    os.makedirs("reports", exist_ok=True)
    df[cols].to_csv("reports/project_scan.csv", index=False)

    shown = df[df["sp_tickets"] >= MIN_PROJECT_SP_TICKETS]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    print("\n" + "=" * 120)
    print(f"PROJECTS WITH >= {MIN_PROJECT_SP_TICKETS} STORY-POINTED TICKETS "
          f"(qualifies = >= {MIN_RESOLVED_WITH_SP} resolved tickets with Story Points)")
    print("=" * 120)
    print(shown[cols].to_string(index=False))
    print(f"\nQualifying projects: {int(df['qualifies'].sum())}")
    print(f"Projects with real estimation error data (actual SP differs from estimate on >= 100 tickets): "
          f"{int((df['actual_differs'] >= 100).sum())}")
    print("Full table saved to reports/project_scan.csv")


if __name__ == "__main__":
    main()
