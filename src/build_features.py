import pandas as pd
import numpy as np
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
collection = client["JiraReposAnon"]["Spring"]

SP_FIELD = "customfield_10142"
EARLY_WINDOW_DAYS = 5

# ---- Load the label/split file we already built ----
labels = pd.read_csv("data/processed/xd_all_storypoints_with_splits.csv")
labels["created"] = pd.to_datetime(labels["created"])
keys = labels["key"].tolist()

# ---- Pull full issue documents (metadata + changelog) for these keys ----
cursor = collection.find(
    {"key": {"$in": keys}},
    {
        "key": 1,
        "fields.issuetype.name": 1,
        "fields.priority.name": 1,
        "fields.components": 1,
        "fields.labels": 1,
        "fields.summary": 1,
        "fields.description": 1,
        "fields.reporter.key": 1,
        "fields.created": 1,
        "changelog.histories": 1,
        "fields.comments": 1,
    },
)

records = []
for doc in cursor:
    f = doc["fields"]
    records.append({
        "key": doc["key"],
        "created": pd.to_datetime(f.get("created")),
        "issuetype": f.get("issuetype", {}).get("name"),
        "priority": f.get("priority", {}).get("name"),
        "num_components": len(f.get("components") or []),
        "num_labels": len(f.get("labels") or []),
        "summary_len": len(f.get("summary") or ""),
        "description_len": len(f.get("description") or ""),
        "reporter": f.get("reporter", {}).get("key"),
        "changelog_histories": doc.get("changelog", {}).get("histories", []),
        "comments": f.get("comments", []),
    })

meta = pd.DataFrame(records)
print(f"Pulled metadata for {len(meta)} issues")

# ---- Reporter history features (strictly prior issues only) ----
meta_sorted = meta.sort_values("created").reset_index(drop=True)
reporter_prev_count = []
for i, row in meta_sorted.iterrows():
    prior = meta_sorted[(meta_sorted["reporter"] == row["reporter"]) & (meta_sorted["created"] < row["created"])]
    reporter_prev_count.append(len(prior))
meta_sorted["reporter_prev_issue_count"] = reporter_prev_count

# ---- Early-window changelog features (capped at EARLY_WINDOW_DAYS) ----
def early_window_features(row):
    cutoff = row["created"] + pd.Timedelta(days=EARLY_WINDOW_DAYS)
    histories = row["changelog_histories"]

    early_status_changes = 0
    early_assignee_changes = 0
    early_priority_changes = 0
    first_status_change_days = None

    for h in histories:
        h_created = pd.to_datetime(h.get("created"))
        if h_created > cutoff:
            continue
        for item in h.get("items", []):
            field = item.get("field")
            if field == "status":
                early_status_changes += 1
                if first_status_change_days is None:
                    first_status_change_days = (h_created - row["created"]).total_seconds() / 86400
            elif field == "assignee":
                early_assignee_changes += 1
            elif field == "priority":
                early_priority_changes += 1

    early_comments = 0
    for c in row["comments"]:
        c_created = pd.to_datetime(c.get("created"))
        if c_created <= cutoff:
            early_comments += 1

    return pd.Series({
        "early_status_changes": early_status_changes,
        "early_assignee_changes": early_assignee_changes,
        "early_priority_changes": early_priority_changes,
        "first_status_change_days": first_status_change_days,
        "early_comment_count": early_comments,
    })

early_feats = meta_sorted.apply(early_window_features, axis=1)
meta_sorted = pd.concat([meta_sorted, early_feats], axis=1)

# ---- Merge with labels/splits, drop raw changelog/comments (not needed downstream) ----
final = labels.merge(
    meta_sorted.drop(columns=["changelog_histories", "comments", "created"]),
    on="key",
    how="left",
)

print("\nFinal feature set preview:")
print(final.head())
print("\nColumns:", list(final.columns))
print("\nMissing values per column:")
print(final.isna().sum())

final.to_csv("data/processed/xd_features.csv", index=False)
print("\nSaved: data/processed/xd_features.csv")