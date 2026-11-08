"""
scan_sprints.py - can "sprint spillover" be the delay label?

Sprint spillover = a ticket that was put into a sprint but not finished in it, so it
was carried over into a later sprint (it appears in 2 or more sprints).

For each project in projects_config.py, over its COMPLETED story-pointed tickets
(same definition as mp_extract.py), this counts:
  - sprint_pct          tickets with any sprint in the Sprint field
  - sprint_history_pct  tickets whose Sprint changes appear in the changelog
  - multi_sprint_pct    tickets that were in 2+ sprints (= spillover rate)
  - duedate_pct         tickets with a due date
and prints a few raw Sprint values so the field format can be checked by eye.

Decision rule (fixed in advance, see the project plan doc):
  sprint spillover becomes the PRIMARY delay label if at least 8 projects have
  sprint information for at least 70% of completed tickets; otherwise the
  long-cycle-time label is primary.

Run from the project root (MongoDB must be running):
    python src/scan_sprints.py
Output: printed table + reports/sprint_scan.csv
"""
import json
import os
import re

import pandas as pd
from pymongo import MongoClient

from projects_config import DONE_STATUSES, PROJECTS, SP_FIELDS, resolution_is_excluded

FIELD_INFO_PATH = os.path.join("data", "raw", "ThePublicJiraDataset", "0. DataDefinition",
                               "jira_field_information.json")
MIN_COVERAGE_PCT = 70
MIN_PROJECTS = 8
N_SAMPLES = 2

client = MongoClient("mongodb://localhost:27017/")
db = client["JiraReposAnon"]
COLLECTIONS = {name.lower(): name for name in db.list_collection_names()}


def sprint_fields(field_info):
    """Sprint field id(s) per repository: fields named exactly 'Sprint' (Jira Software's sprint field)."""
    out = {}
    for repo, fields in field_info.items():
        ids = [f["id"] for f in fields if str(f.get("name", "")).strip().lower() == "sprint"]
        if ids:
            out[repo.lower()] = ids
    return out


def count_sprints(value):
    """
    Number of sprints in a Sprint field value. Handles the formats Jira exports use:
    a list of strings like 'com.atlassian.greenhopper.service.sprint.Sprint@1a2b[id=12,...,name=Sprint 4,...]',
    a list of dicts/ids, or a single string / comma-separated names.
    """
    if value is None:
        return 0
    if isinstance(value, list):
        return len([v for v in value if v not in (None, "", [])])
    if isinstance(value, dict):
        return 1
    text = str(value).strip()
    if not text:
        return 0
    ids = re.findall(r"\bid=\d+", text)
    if ids:
        return len(set(ids))
    return len([part for part in text.split(",") if part.strip()])


def sprints_from_changelog(histories):
    """(any Sprint change in the changelog?, number of distinct sprints the ticket was ever added to)."""
    seen, any_change = set(), False
    for h in histories or []:
        for item in h.get("items") or []:
            if str(item.get("field", "")).strip().lower() != "sprint":
                continue
            any_change = True
            for key in ("to", "toString"):
                raw = item.get(key)
                if raw:
                    for part in str(raw).split(","):
                        if part.strip():
                            seen.add(part.strip())
                    break
    return any_change, len(seen)


def scan_project(project, sprint_ids):
    repo, key = project["repo"], project["key"]
    coll = db[COLLECTIONS[repo.lower()]]
    sp_ids = SP_FIELDS[repo]
    projection = {"changelog.histories.items": 1, "fields.resolutiondate": 1, "fields.status.name": 1,
                  "fields.resolution.name": 1, "fields.duedate": 1}
    for fid in sp_ids + sprint_ids:
        projection[f"fields.{fid}"] = 1
    query = {"fields.project.key": key, "$or": [{f"fields.{fid}": {"$ne": None}} for fid in sp_ids]}

    n = with_sprint = with_history = multi = with_due = 0
    samples = []
    for doc in coll.find(query, projection):
        f = doc.get("fields", {}) or {}
        status = str((f.get("status") or {}).get("name") or "").lower()
        resolution = (f.get("resolution") or {}).get("name")
        if not f.get("resolutiondate") or status not in DONE_STATUSES or resolution_is_excluded(resolution):
            continue
        n += 1
        field_count = max([count_sprints(f.get(fid)) for fid in sprint_ids] or [0])
        changed, log_count = sprints_from_changelog((doc.get("changelog") or {}).get("histories"))
        sprints = max(field_count, log_count)
        with_sprint += sprints > 0
        with_history += changed
        multi += sprints >= 2
        with_due += bool(f.get("duedate"))
        if field_count and len(samples) < N_SAMPLES:
            raw = next(f.get(fid) for fid in sprint_ids if count_sprints(f.get(fid)))
            samples.append(str(raw)[:160])

    pct = lambda x: round(100 * x / n, 1) if n else 0.0
    return {
        "repository": repo, "project": key, "docs_project": project["docs"],
        "sprint_field_ids": ",".join(sprint_ids) or "none",
        "completed_tickets": n,
        "sprint_pct": pct(with_sprint),
        "sprint_history_pct": pct(with_history),
        "multi_sprint_pct": pct(multi),
        "multi_sprint_tickets": multi,
        "duedate_pct": pct(with_due),
    }, samples


def main():
    with open(FIELD_INFO_PATH, encoding="utf-8") as f:
        sprint_by_repo = sprint_fields(json.load(f))

    rows = []
    for p in PROJECTS:
        ids = sprint_by_repo.get(p["repo"].lower(), [])
        print(f"\n=== {p['repo']} / {p['key']} (Sprint field: {ids or 'none found'}) ===")
        try:
            row, samples = scan_project(p, ids)
        except Exception as exc:
            print(f"  !! failed: {exc}")
            continue
        rows.append(row)
        print(f"  completed {row['completed_tickets']} | sprint info {row['sprint_pct']}% | "
              f"sprint changelog {row['sprint_history_pct']}% | in 2+ sprints {row['multi_sprint_pct']}% "
              f"({row['multi_sprint_tickets']}) | due date {row['duedate_pct']}%")
        for s in samples:
            print(f"    sample Sprint value: {s}")

    df = pd.DataFrame(rows)
    os.makedirs("reports", exist_ok=True)
    df.to_csv("reports/sprint_scan.csv", index=False)

    qualifying = df[df["sprint_pct"] >= MIN_COVERAGE_PCT]
    pd.set_option("display.width", 220)
    print("\n" + "=" * 110)
    print(df.drop(columns=["sprint_field_ids"]).to_string(index=False))
    print("\n" + "=" * 110)
    print(f"Projects with sprint info for >= {MIN_COVERAGE_PCT}% of completed tickets: {len(qualifying)} "
          f"({', '.join(qualifying['project']) or 'none'})")
    if len(qualifying) >= MIN_PROJECTS:
        print(f"RULE MET (>= {MIN_PROJECTS} projects): sprint spillover becomes the PRIMARY delay label.")
    else:
        print(f"RULE NOT MET (< {MIN_PROJECTS} projects): long cycle time becomes the PRIMARY delay label; "
              "sprint spillover can still be a secondary analysis on the qualifying projects.")
    print("Saved reports/sprint_scan.csv")


if __name__ == "__main__":
    main()
