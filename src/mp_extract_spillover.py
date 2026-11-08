"""
mp_extract_spillover.py - build the SPRINT-SPILLOVER dataset (primary delay label).

Delay label (contains no Story Points):
    spilled = 1 if the ticket was NOT completed by the end of the FIRST sprint it was
              committed to (completed = resolution date + done-like status + a resolution
              that is not Won't Fix / Duplicate / ...); otherwise 0.
    A ticket still unfinished when its first sprint closed is KNOWN to have spilled, so
    unfinished tickets are labelled too (no censoring bias). Tickets whose first sprint
    had not closed when the data was collected are left out (outcome unknown).

Prediction point = COMMITMENT TIME t0 = when the ticket first sits in a started sprint
    (max(sprint start, time the ticket was added to it)). Everything a feature uses is
    rebuilt AS OF t0, so nothing from after the commitment leaks in:
      - sp_team_commit      the team's Story Points as they stood at t0 (changelog walked back)
      - early_*             activity BEFORE commitment (status / assignee / priority changes,
                            comments between creation and t0) - same column names as the
                            day-5 pipeline so the shared code can be reused
      - first_status_change_days_imputed  days from creation to the first status change
                            before t0 (no change -> backlog_days)
      - backlog_days        days from creation to commitment
      - sprint_length_days  planned length of the committed sprint (known at planning)
Split: chronological 70/15/15 per project, by commitment time.

Sprint membership is read from the Sprint field (final value; Jira keeps closed sprints in
it) and from the Sprint changes in the changelog (covers tickets moved out of a sprint).
Sprint dates come from a per-project sprint calendar built from every ticket's Sprint field.

Run from the project root (MongoDB must be running):
    python src/mp_extract_spillover.py            # the 13 qualifying projects
    python src/mp_extract_spillover.py XD EVG     # some projects only
Output:
    data/processed/mp_spillover_dataset.csv
    reports/mp_spillover_extraction_summary.csv
"""
import os
import re
import sys

import numpy as np
import pandas as pd

from mp_extract import (COLLECTIONS, chronological_split, db, first_number, parse_histories,
                        story_points_as_of, to_utc)
from projects_config import (DONE_STATUSES, PROJECTS, SP_FIELDS, SP_MAX, SPILLOVER_PROJECTS,
                             SPRINT_FIELDS, resolution_is_excluded)

INF = pd.Timestamp.max.tz_localize("UTC")


# -----------------------------------------------------------------------------
# Sprint field parsing
# -----------------------------------------------------------------------------
def _get(pattern, text):
    m = re.search(pattern, text)
    if not m:
        return None
    value = m.group(1).strip()
    return None if value in ("", "<null>", "null", "None") else value


def parse_sprint_value(value):
    """Sprint field value -> list of {id, state, name, start, end, complete}. Handles Jira's
    'com.atlassian.greenhopper...Sprint@x[id=..,state=..,name=..,startDate=..,...]' strings and dicts."""
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    out = []
    for item in items:
        if isinstance(item, dict):
            if item.get("id") is None:
                continue
            out.append({"id": int(item["id"]), "state": str(item.get("state") or "").upper(),
                        "name": item.get("name"), "start": to_utc(item.get("startDate")),
                        "end": to_utc(item.get("endDate")), "complete": to_utc(item.get("completeDate"))})
            continue
        text = str(item)
        blocks = re.findall(r"\[([^\[\]]*id=\d+[^\[\]]*)\]", text) or [text]
        for b in blocks:
            sid = _get(r"\bid=(\d+)", b)
            if sid is None:
                continue
            out.append({"id": int(sid), "state": (_get(r"\bstate=([A-Za-z]+)", b) or "").upper(),
                        "name": _get(r"\bname=(.*?),(?:goal|startDate|endDate|completeDate|sequence)=", b),
                        "start": to_utc(_get(r"\bstartDate=([^,\]]+)", b)),
                        "end": to_utc(_get(r"\bendDate=([^,\]]+)", b)),
                        "complete": to_utc(_get(r"\bcompleteDate=([^,\]]+)", b))})
    return out


def sprint_ids_from_change(raw_ids, raw_names, name_to_id):
    """Sprint ids in one changelog value ('5520, 5521'); falls back to sprint names."""
    if raw_ids not in (None, ""):
        return {int(x) for x in re.findall(r"\d+", str(raw_ids))}
    if raw_names not in (None, ""):
        return {name_to_id[n.strip()] for n in str(raw_names).split(",") if n.strip() in name_to_id}
    return set()


def membership_intervals(created, field_ids, parsed_histories, name_to_id):
    """[(sprint_id, member_from, member_until)] from the changelog, else from the final field."""
    events = []
    for t, h in parsed_histories:
        for item in h.get("items") or []:
            if str(item.get("field", "")).strip().lower() == "sprint":
                events.append((t, sprint_ids_from_change(item.get("from"), item.get("fromString"), name_to_id),
                               sprint_ids_from_change(item.get("to"), item.get("toString"), name_to_id)))
    if not events:
        return [(sid, created, INF) for sid in field_ids]

    current = set(events[0][1])                 # sprints the ticket had before its first sprint change
    since = {sid: created for sid in current}
    intervals = []
    for t, _, to_ids in events:
        for sid in current - to_ids:
            intervals.append((sid, since.pop(sid, created), t))
        for sid in to_ids - current:
            since[sid] = t
        current = set(to_ids)
    intervals += [(sid, since.get(sid, created), INF) for sid in current]
    logged = {i[0] for i in intervals}
    intervals += [(sid, created, INF) for sid in field_ids if sid not in logged]   # field-only sprints
    return intervals


def first_commitment(intervals, calendar):
    """Earliest started sprint the ticket was a member of -> (sprint, commitment time t0) or (None, None)."""
    best = None
    for sid, a, b in intervals:
        s = calendar.get(sid)
        if s is None or s["start"] is None:
            continue
        end_eff = s["complete"] or s["end"] or INF
        if a < end_eff and b > s["start"]:              # was in the sprint while it ran
            t0 = max(a, s["start"])
            if best is None or t0 < best[1]:
                best = (s, t0)
    return best if best else (None, None)


def activity_before(created, cutoff, parsed_histories, comments):
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
    return status, assignee, priority, n_comments, first_status


# -----------------------------------------------------------------------------
# one project
# -----------------------------------------------------------------------------
def extract_project(project):
    repo, key = project["repo"], project["key"]
    coll = db[COLLECTIONS[repo.lower()]]
    sp_ids, sprint_ids = SP_FIELDS[repo], SPRINT_FIELDS[repo]

    projection = {"key": 1, "changelog.histories": 1, "fields.created": 1, "fields.resolutiondate": 1,
                  "fields.status.name": 1, "fields.resolution.name": 1, "fields.issuetype.name": 1,
                  "fields.priority.name": 1, "fields.components": 1, "fields.labels": 1, "fields.summary": 1,
                  "fields.description": 1, "fields.reporter": 1, "fields.comments": 1, "fields.comment": 1}
    for fid in sp_ids + sprint_ids:
        projection[f"fields.{fid}"] = 1
    query = {"fields.project.key": key, "$or": [{f"fields.{fid}": {"$ne": None}} for fid in sp_ids]}

    # pass 1: load tickets and build the project's sprint calendar
    tickets, calendar = [], {}
    for doc in coll.find(query, projection):
        f = doc.get("fields", {}) or {}
        sp, created = first_number(f, sp_ids), to_utc(f.get("created"))
        if sp is None or created is None:
            continue
        sprints = [s for fid in sprint_ids for s in parse_sprint_value(f.get(fid))]
        for s in sprints:
            known = calendar.get(s["id"])
            if known is None or (known["start"] is None and s["start"] is not None):
                calendar[s["id"]] = s
        tickets.append((doc, f, sp, created, [s["id"] for s in sprints]))
    name_to_id = {s["name"]: sid for sid, s in calendar.items() if s.get("name")}

    # pass 2: commitment, label, features as of commitment
    diag = {"project": key, "repository": repo, "tickets_with_sp": len(tickets), "sprints_in_calendar": len(calendar)}
    counts = {"outlier": 0, "never_committed": 0, "first_sprint_not_closed": 0,
              "excluded_resolution": 0, "resolved_before_commit": 0}
    rows = []
    for doc, f, sp, created, field_ids in tickets:
        if sp < 0 or sp > SP_MAX:
            counts["outlier"] += 1
            continue
        histories = parse_histories((doc.get("changelog") or {}).get("histories", []))
        sprint, t0 = first_commitment(membership_intervals(created, field_ids, histories, name_to_id), calendar)
        if sprint is None:
            counts["never_committed"] += 1
            continue
        sprint_end = sprint["complete"] or sprint["end"]
        if sprint["state"] != "CLOSED" or sprint_end is None:
            counts["first_sprint_not_closed"] += 1
            continue
        resolution = (f.get("resolution") or {}).get("name")
        resolved_at = to_utc(f.get("resolutiondate"))
        if resolved_at is not None and resolution_is_excluded(resolution):
            counts["excluded_resolution"] += 1
            continue
        done_status = str((f.get("status") or {}).get("name") or "").lower() in DONE_STATUSES
        completed_at = resolved_at if (resolved_at is not None and done_status) else None
        if completed_at is not None and completed_at < t0:
            counts["resolved_before_commit"] += 1
            continue

        sp_commit, _, sp_changed_after = story_points_as_of(sp, histories, t0, sp_ids)
        comments = f.get("comments")
        if comments is None:
            comments = (f.get("comment") or {}).get("comments", [])
        status, assignee, prio, n_comments, first_status = activity_before(created, t0, histories, comments)
        backlog_days = (t0 - created).total_seconds() / 86400
        summary, description = f.get("summary") or "", f.get("description") or ""
        reporter = f.get("reporter") or {}
        rows.append({
            "project": key, "repository": repo, "key": doc.get("key"),
            "created": created, "commit_time": t0, "first_sprint_id": sprint["id"],
            "first_sprint_end": sprint_end, "completed_at": completed_at,
            "spilled": int(completed_at is None or completed_at > sprint_end),
            "story_points": sp, "sp_team_commit": sp_commit, "sp_changed_after_commit": int(sp_changed_after),
            "issuetype": (f.get("issuetype") or {}).get("name") or "Unknown",
            "priority": (f.get("priority") or {}).get("name") or "None",
            "num_components": len(f.get("components") or []), "num_labels": len(f.get("labels") or []),
            "summary": summary, "description": description,
            "summary_len": len(summary), "description_len": len(description),
            "has_description": int(len(description) > 0),
            "reporter": reporter.get("key") or reporter.get("name") or reporter.get("accountId"),
            "early_status_changes": status, "early_assignee_changes": assignee,
            "early_priority_changes": prio, "early_comment_count": n_comments,
            "first_status_change_days_imputed": first_status if first_status is not None else backlog_days,
            "backlog_days": backlog_days,
            "sprint_length_days": ((sprint["end"] or sprint_end) - sprint["start"]).total_seconds() / 86400,
        })

    diag.update({f"dropped_{k}": v for k, v in counts.items()})
    df = pd.DataFrame(rows)
    diag["labelled"] = len(df)
    if df.empty:
        return df, diag

    # chronological split by COMMITMENT time (chronological_split sorts on 'created')
    df = df.rename(columns={"created": "created_at"}).rename(columns={"commit_time": "created"})
    df = chronological_split(df)
    df = df.rename(columns={"created": "commit_time"}).rename(columns={"created_at": "created"})

    df["reporter_prev_issue_count"] = (
        df.groupby(df["reporter"].fillna("__none__"))["created"].rank(method="min").astype(int) - 1)
    df.loc[df["reporter"].isna(), "reporter_prev_issue_count"] = 0
    df["eligible_for_risk_model"] = 1

    for split in ("train", "val", "test"):
        s = df[df["split"] == split]
        diag[f"n_{split}"] = len(s)
        diag[f"spill_rate_{split}"] = round(float(s["spilled"].mean()), 3) if len(s) else None
    diag["sp_known_at_commit_pct"] = round(100 * df["sp_team_commit"].notna().mean(), 1)
    diag["sp_changed_after_commit_pct"] = round(100 * df["sp_changed_after_commit"].mean(), 1)
    diag["unfinished_but_labelled"] = int(df["completed_at"].isna().sum())
    diag["median_backlog_days"] = round(float(df["backlog_days"].median()), 1)
    diag["median_sprint_length_days"] = round(float(df["sprint_length_days"].median()), 1)
    return df, diag


def main():
    only = {k.upper() for k in sys.argv[1:]}
    keys = [k for k in SPILLOVER_PROJECTS if not only or k in only]
    projects = [p for p in PROJECTS if p["key"] in keys]

    frames, diags = [], []
    for p in projects:
        print(f"\n=== {p['repo']} / {p['key']} ===")
        try:
            df, diag = extract_project(p)
        except Exception as exc:
            print(f"  !! failed: {exc}")
            continue
        diags.append(diag)
        if df.empty:
            print("  no labelled tickets")
            continue
        frames.append(df)
        print(f"  tickets {diag['tickets_with_sp']} | labelled {diag['labelled']} | never in a sprint "
              f"{diag['dropped_never_committed']} | first sprint not closed {diag['dropped_first_sprint_not_closed']} | "
              f"excluded resolutions {diag['dropped_excluded_resolution']}")
        print(f"  train/val/test {diag['n_train']}/{diag['n_val']}/{diag['n_test']} | spill rate "
              f"{diag['spill_rate_train']}/{diag['spill_rate_val']}/{diag['spill_rate_test']} | "
              f"unfinished but labelled {diag['unfinished_but_labelled']}")
        print(f"  SP known at commitment {diag['sp_known_at_commit_pct']}% | changed after commitment "
              f"{diag['sp_changed_after_commit_pct']}% | median backlog {diag['median_backlog_days']} days | "
              f"median sprint {diag['median_sprint_length_days']} days")

    if not frames:
        print("Nothing extracted.")
        return
    out = pd.concat(frames, ignore_index=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    out.drop(columns=["reporter"]).to_csv("data/processed/mp_spillover_dataset.csv", index=False)
    summary = pd.DataFrame(diags)
    summary.to_csv("reports/mp_spillover_extraction_summary.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n" + "=" * 110)
    print(summary[["project", "labelled", "n_train", "n_val", "n_test", "spill_rate_train", "spill_rate_test",
                   "sp_known_at_commit_pct", "sp_changed_after_commit_pct"]].to_string(index=False))
    print(f"\nSaved data/processed/mp_spillover_dataset.csv ({len(out)} tickets, {len(frames)} projects)")
    print("Saved reports/mp_spillover_extraction_summary.csv")


if __name__ == "__main__":
    main()
