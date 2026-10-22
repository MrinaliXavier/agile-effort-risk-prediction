import pandas as pd
import numpy as np
from pymongo import MongoClient

EARLY_WINDOW_DAYS = 5  # must match the window used in build_features.py


def prepare_dataset():
    print("--- Step 1: Loading feature set and risk candidates ---")
    feat_df = pd.read_csv("data/processed/xd_features.csv")
    done_df = pd.read_csv("data/processed/xd_done_with_risk_candidates.csv")

    print(f"Features loaded: {len(feat_df)} rows")
    print(f"Done issues with risk labels loaded: {len(done_df)} rows")

    # --- Step 2: Retrieve raw text (summary, description) from MongoDB ---
    print("\n--- Step 2: Retrieving summary & description text from MongoDB ---")
    client = MongoClient("mongodb://localhost:27017/")
    collection = client["JiraReposAnon"]["Spring"]

    keys = feat_df["key"].tolist()
    cursor = collection.find(
        {"key": {"$in": keys}},
        {"key": 1, "fields.summary": 1, "fields.description": 1}
    )

    text_records = []
    for doc in cursor:
        f = doc.get("fields", {})
        text_records.append({
            "key": doc["key"],
            "summary": f.get("summary") or "",
            "description": f.get("description") or "",
        })
    text_df = pd.DataFrame(text_records)
    print(f"Retrieved text for {len(text_df)} issues")

    feat_df = feat_df.merge(text_df, on="key", how="left")
    feat_df["summary"] = feat_df["summary"].fillna("")
    feat_df["description"] = feat_df["description"].fillna("")

    # --- Step 3: Merge Risk Labels ---
    print("\n--- Step 3: Merging risk labels onto feature set ---")
    risk_cols = ["key", "cycle_time_days", "residual", "risk_20pct", "risk_25pct", "risk_30pct"]
    merged = feat_df.merge(done_df[risk_cols], on="key", how="left")

    # --- Step 4: Feature Engineering & Missing Value Handling ---
    print("\n--- Step 4: Engineering features and handling missing values ---")

    merged["had_early_status_change"] = merged["first_status_change_days"].notna().astype(int)
    merged["first_status_change_days_imputed"] = merged["first_status_change_days"].fillna(EARLY_WINDOW_DAYS)

    priority_map = {"Trivial": 1, "Minor": 2, "Major": 3, "Critical": 4, "Blocker": 5}
    merged["priority_ordinal"] = merged["priority"].map(priority_map).fillna(2).astype(int)

    merged["has_description"] = (merged["description_len"] > 0).astype(int)
    merged["has_components"] = (merged["num_components"] > 0).astype(int)
    merged["has_labels"] = (merged["num_labels"] > 0).astype(int)

    # --- Step 4b. FIX: flag which Done issues are actually eligible for risk modelling ---
    # An issue that already resolved WITHIN the early observation window has a risk label
    # that is trivially determined by the fact it's already Done (100% "on track" by
    # construction), and its early-window changelog features partially encode that same
    # completion event. Including these rows teaches the model to read "already finished"
    # off the changelog instead of learning genuine early risk signals.
    #
    # eligible_for_risk_model = 1 only for Done issues that were STILL OPEN past the
    # observation window (cycle_time_days > EARLY_WINDOW_DAYS) - i.e. issues where a
    # risk prediction at the window cutoff is a genuine forecast, not a foregone conclusion.
    merged["eligible_for_risk_model"] = np.where(
        merged["cycle_time_days"].notna() & (merged["cycle_time_days"] > EARLY_WINDOW_DAYS),
        1,
        0,
    )

    # --- Step 5: Summary of Dataset ---
    print("\nDataset dimensions:", merged.shape)
    print("\nSplit breakdown (all issues for Effort task):")
    print(merged["split"].value_counts())

    done_subset = merged[merged["risk_25pct"].notna()]
    n_total_done = len(done_subset)
    n_resolved_in_window = (done_subset["cycle_time_days"] <= EARLY_WINDOW_DAYS).sum()
    print(f"\nDone issues: {n_total_done}")
    print(f"  Resolved WITHIN the {EARLY_WINDOW_DAYS}-day window (excluded from risk modelling): "
          f"{n_resolved_in_window} ({n_resolved_in_window / n_total_done:.1%})")

    eligible = merged[merged["eligible_for_risk_model"] == 1]
    print(f"  Eligible for risk modelling (still open past day {EARLY_WINDOW_DAYS}): {len(eligible)}")
    print("\nSplit breakdown for ELIGIBLE Done issues only (Risk task, leak-safe population):")
    print(eligible.groupby("split")["risk_25pct"].value_counts(normalize=True).unstack())

    output_path = "data/processed/xd_modeling_dataset.csv"
    merged.to_csv(output_path, index=False)
    print(f"\nSuccessfully generated and saved unified dataset to: {output_path}")


if __name__ == "__main__":
    prepare_dataset()