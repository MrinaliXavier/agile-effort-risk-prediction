import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.linear_model import LinearRegression

client = MongoClient("mongodb://localhost:27017/")
collection = client["JiraReposAnon"]["Spring"]

SP_FIELD = "customfield_10142"

# ---- 1. Pull XD issues with Story Points populated ----
cursor = collection.find(
    {"fields.project.key": "XD", f"fields.{SP_FIELD}": {"$ne": None}},
    {
        "key": 1,
        "fields.created": 1,
        "fields.resolutiondate": 1,
        "fields.status.name": 1,
        f"fields.{SP_FIELD}": 1,
    },
)

rows = []
for doc in cursor:
    f = doc["fields"]
    rows.append({
        "key": doc["key"],
        "created": f.get("created"),
        "resolutiondate": f.get("resolutiondate"),
        "status": f.get("status", {}).get("name"),
        "story_points": f.get(SP_FIELD),
    })

df = pd.DataFrame(rows)
df["created"] = pd.to_datetime(df["created"])
df["resolutiondate"] = pd.to_datetime(df["resolutiondate"])

print(f"Total issues with Story Points (before cleaning): {len(df)}")

# ---- 2. Drop implausible outliers ----
# Spotted earlier: XD-2602 has story_points = 2567, a clear data-entry error.
# Use a sane upper cap (real XD values topped out at 100 otherwise).
before = len(df)
df = df[df["story_points"] <= 200].copy()
print(f"Dropped {before - len(df)} outlier row(s) with implausible Story Points values.")

# ---- 3. Chronological split (70/15/15) ----
df = df.sort_values("created").reset_index(drop=True)
n = len(df)
train_cut = df.loc[int(n * 0.70), "created"]
val_cut = df.loc[int(n * 0.85), "created"]

df["split"] = pd.cut(
    df["created"],
    bins=[df["created"].min() - pd.Timedelta(days=1), train_cut, val_cut, df["created"].max() + pd.Timedelta(days=1)],
    labels=["train", "val", "test"],
)

print("\nSplit sizes:")
print(df["split"].value_counts().sort_index())

# ---- 4. Done issues -> cycle time ----
done = df[(df["status"] == "Done") & (df["resolutiondate"].notna())].copy()
done["cycle_time_days"] = (done["resolutiondate"] - done["created"]).dt.total_seconds() / 86400
done = done[done["cycle_time_days"] > 0]  # safety: drop any non-positive durations

print(f"\nDone issues usable for risk label: {len(done)}")

# ---- 5. Fit log-linear regression on TRAIN ONLY ----
# cycle time is right-skewed, so we model log(cycle_time) ~ story_points
train_done = done[done["split"] == "train"].copy()
train_done["log_cycle_time"] = np.log1p(train_done["cycle_time_days"])  # log1p handles any 0s safely

reg = LinearRegression()
X_train = train_done[["story_points"]].values
y_train = train_done["log_cycle_time"].values
reg.fit(X_train, y_train)

print(f"\nFitted regression: log(cycle_time+1) = {reg.intercept_:.3f} + {reg.coef_[0]:.3f} * story_points")

# ---- 6. Compute residuals for ALL done issues, using the train-fitted line ----
done["log_cycle_time"] = np.log1p(done["cycle_time_days"])
done["predicted_log_cycle_time"] = reg.predict(done[["story_points"]].values)
done["residual"] = done["log_cycle_time"] - done["predicted_log_cycle_time"]

# ---- 7. Try a few quantile cutoffs, computed on TRAIN residuals only ----
train_residuals = done[done["split"] == "train"]["residual"]

for top_pct in [0.20, 0.25, 0.30]:
    cutoff = train_residuals.quantile(1 - top_pct)
    done[f"risk_{int(top_pct*100)}pct"] = (done["residual"] > cutoff).astype(int)
    print(f"\n--- Top {int(top_pct*100)}% cutoff (residual > {cutoff:.3f}) ---")
    for split in ["train", "val", "test"]:
        subset = done[done["split"] == split]
        if len(subset) == 0:
            continue
        counts = subset[f"risk_{int(top_pct*100)}pct"].value_counts().sort_index()
        balance = subset[f"risk_{int(top_pct*100)}pct"].value_counts(normalize=True).sort_index()
        print(f"  {split}: n={len(subset)}  on-track={counts.get(0,0)} ({balance.get(0,0):.1%})  at-risk={counts.get(1,0)} ({balance.get(1,0):.1%})")

# ---- 8. Save outputs ----
done.to_csv("data/processed/xd_done_with_risk_candidates.csv", index=False)
df.to_csv("data/processed/xd_all_storypoints_with_splits.csv", index=False)
print("\nSaved updated label files to data/processed/")