"""
mp_explain.py - per-ticket and global SHAP explanations of the delay-risk pipeline (thesis output).

For each project, the SAME models the evaluation selected on validation
(reports/mp_<label>_results.json: Stage 1 effort model + Stage 2 family, proposed variant
'+ Team SP + Model SP') are refitted on train+val and explained on the TEST tickets, so every
explanation is for a ticket the models never saw.

Outputs (reports/explanations/):
  <PROJECT>_examples.txt        three worked tickets in plain words: the highest-risk ticket,
                                the lowest-risk ticket, and the ticket the model thinks is most
                                UNDER-estimated by the team (model estimate / team estimate largest)
  <PROJECT>_examples.json       the same, as data
  <PROJECT>_global_importance.csv  share of the model's total |SHAP| per feature group on the test set
  cross_project_importance.csv  feature groups x projects, plus mean share and mean rank

Run from the project root, after mp_train_evaluate.py:
    python src/mp_explain.py                  # spillover label, all evaluated projects
    python src/mp_explain.py XD EVG           # some projects only
    python src/mp_explain.py --label residual
"""
import json
import os
import sys

import numpy as np
import pandas as pd

from common import (CAT_FEATURES, PROPOSED_VARIANT, RANDOM_SEED, add_full_text, build_risk_pipeline,
                    make_risk_variants, most_common_baselines, to_dense)
from explain import explain_ticket, global_importance
from mp_train_evaluate import LABELS, add_stage1_predictions, population

N_BACKGROUND = 200
OUT_DIR = "reports/explanations"


def background(pipe, df):
    sample = df.sample(n=min(N_BACKGROUND, len(df)), random_state=RANDOM_SEED)
    return to_dense(pipe.named_steps["preprocessor"].transform(sample))


def explain_project(df, key, sel, cfg):
    label, team_col = cfg["label"], cfg["team_col"]
    variants = make_risk_variants(cfg["base"])
    df = add_full_text(df)
    train = df[df["split"] == "train"].reset_index(drop=True)
    val = df[df["split"] == "val"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    trainval = pd.concat([train, val], ignore_index=True)
    baselines = most_common_baselines(train)

    tv_e, te_e, stage1 = add_stage1_predictions(trainval, test, sel["effort_model"], baselines, team_col)
    tv_done, te_done = population(tv_e, label), population(te_e, label)
    stage2 = build_risk_pipeline(sel["risk_family"], cat_baselines=baselines, num_features=variants[PROPOSED_VARIANT])
    stage2.fit(tv_done, tv_done[label].values)
    bg1, bg2 = background(stage1, trainval), background(stage2, tv_done)
    base_rate = float(tv_done[label].mean())

    # global importance on the test set
    imp = global_importance(stage2, bg2, te_done)
    imp.insert(0, "project", key)
    imp.to_csv(f"{OUT_DIR}/{key}_global_importance.csv", index=False)

    # three worked examples
    probs = stage2.predict_proba(te_done)[:, 1]
    te_done = te_done.assign(_prob=probs)
    known = te_done[(te_done["sp_missing_team"] == 0) & (te_done[team_col] >= 1)]   # 0-point commitments are not comparable
    picks = {"highest predicted risk": te_done["_prob"].idxmax(),
             "lowest predicted risk": te_done["_prob"].idxmin()}
    if len(known):
        ratio = (known["pred_effort"] + 1) / (known[team_col] + 1)
        picks["model thinks most under-estimated by the team"] = ratio.idxmax()

    texts, records = [f"PROJECT {key} - Stage 1: {sel['effort_model']} | Stage 2: {sel['risk_family']} "
                      f"({PROPOSED_VARIANT}) | label: {label}\n"], []
    for title, idx in picks.items():
        frame = te_done.loc[[idx]].drop(columns=["_prob"])
        result, text = explain_ticket(frame, stage1, stage2, bg1, bg2, project_spill_rate=base_rate)
        outcome = int(frame[label].iloc[0])
        result.update({"example": title, "actual_outcome": outcome})
        records.append(result)
        texts.append(f"--- {title} ---\n{text}\n  ACTUAL OUTCOME: {'missed its sprint' if outcome else 'finished in its sprint'}\n")

    with open(f"{OUT_DIR}/{key}_examples.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(texts))
    with open(f"{OUT_DIR}/{key}_examples.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, default=str)
    return imp, "\n".join(texts)


def main():
    args = sys.argv[1:]
    label_name = "spillover"
    if "--label" in args:
        i = args.index("--label")
        label_name, args = args[i + 1], args[:i] + args[i + 2:]
    cfg = LABELS[label_name]
    results_path = f"reports/mp_{label_name}_results.json"
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"{results_path} not found - run src/mp_train_evaluate.py first.")
    with open(results_path) as f:
        selection = {r["project"]: r for r in json.load(f)["per_project"] if "risk_family" in r}

    data = pd.read_csv(cfg["dataset"])
    data[CAT_FEATURES] = data[CAT_FEATURES].fillna("None").astype(str)
    only = {k.upper() for k in args}
    os.makedirs(OUT_DIR, exist_ok=True)

    importances = []
    for key, sel in selection.items():
        if only and key not in only:
            continue
        print(f"\n=== {key} ===")
        try:
            imp, text = explain_project(data[data["project"] == key].copy(), key, sel, cfg)
        except Exception as exc:
            print(f"  !! failed: {exc}")
            continue
        importances.append(imp)
        print(text)

    if not importances:
        print("Nothing explained.")
        return
    allimp = pd.concat(importances)
    allimp["rank"] = allimp.groupby("project")["share"].rank(ascending=False, method="min")
    table = allimp.pivot_table(index="group", columns="project", values="share")
    table["mean_share"] = table.mean(axis=1)
    table["mean_rank"] = allimp.groupby("group")["rank"].mean()
    table["top3_in_projects"] = allimp[allimp["rank"] <= 3].groupby("group").size()
    table = table.sort_values("mean_share", ascending=False)
    table.round(4).to_csv(f"{OUT_DIR}/cross_project_importance.csv")

    pd.set_option("display.width", 200)
    print("\n" + "=" * 100)
    print("WHAT DRIVES DELAY RISK ACROSS PROJECTS (share of total |SHAP| on test sets)")
    print("=" * 100)
    print(table[["mean_share", "mean_rank", "top3_in_projects"]].fillna(0).round(3).to_string())
    print(f"\nSaved per-project examples and importance to {OUT_DIR}/")


if __name__ == "__main__":
    main()
