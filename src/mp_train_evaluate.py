"""
mp_train_evaluate.py - do effort estimates help predict delivery delay, across projects?

Research question
    Do effort estimates - the team's commitment and an independent ML estimate - help
    predict whether a ticket will miss its delivery commitment?

Hypotheses (test set comparisons, delta = AUC(variant) - AUC(baseline)):
    H1  + Team SP              vs No Effort   the team's estimate carries delay information
    H2  + Team SP + Model SP   vs + Team SP   the model's estimate (= the estimation gap, since
                                              gap = log team SP - log model SP) adds information
                                              BEYOND the team's estimate        <- core claim
    H3  same as H2, on test tickets the team had NOT estimated at prediction time
    plus: + Pred Effort vs No Effort (the original design), + Estimation Gap vs No Effort,
    and a LEAKAGE CHECK: final Story Points instead of Story Points at prediction time.

Labels (choose with --label):
    spillover (default, PRIMARY): not completed by the end of the first committed sprint;
        prediction at sprint commitment.  Dataset: data/processed/mp_spillover_dataset.csv
    residual (SECONDARY): top-25% cycle-time residual vs Story Points; prediction at day 5.
        Its label is partly built from Story Points - see the project plan doc.
        Dataset: data/processed/mp_modeling_dataset.csv

Per project: Stage 1 effort model picked on VALIDATION MAE; Stage 2 family picked on
VALIDATION AUC of the proposed variant; refit on train+val, scored ONCE on test with paired
bootstrap 95% CIs. A project is only analysed if its test set holds >= MIN_CLASS_COUNT
tickets of EACH class (rule fixed in advance: AUC on a handful of delayed tickets is noise).
Across projects: projects improved, significant +/-, median delta, Wilcoxon signed-rank test.

Run from the project root:
    python src/mp_train_evaluate.py                       # spillover, all projects
    python src/mp_train_evaluate.py --label residual      # secondary label
    python src/mp_train_evaluate.py XD EVG                # some projects only
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.dummy import DummyRegressor
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from common import (CAT_FEATURES, EFFORT_CANDIDATES, LEAKAGE_VARIANT, PROPOSED_VARIANT, RANDOM_SEED,
                    RISK_CANDIDATES, STAGE1_NUM_FEATURES, STAGE2_BASE_NUM_FEATURES, add_estimate_features,
                    add_full_text, build_effort_pipeline, build_preprocessor, build_risk_pipeline,
                    eval_effort_metrics, eval_risk_metrics, make_risk_variants, most_common_baselines,
                    oof_effort_predictions, predict_effort)

LABELS = {
    "spillover": {"dataset": "data/processed/mp_spillover_dataset.csv", "label": "spilled",
                  "team_col": "sp_team_commit",
                  "base": STAGE2_BASE_NUM_FEATURES + ["backlog_days", "sprint_length_days"]},
    "residual": {"dataset": "data/processed/mp_modeling_dataset.csv", "label": "risk_25pct",
                 "team_col": "sp_team_day5", "base": STAGE2_BASE_NUM_FEATURES},
}
N_BOOTSTRAP = 1000
MIN_CLASS_COUNT = 10
COMPARISONS = [  # (name, variant, baseline)
    ("H1 team estimate", "+ Team SP", "No Effort"),
    ("H2 model estimate beyond team", PROPOSED_VARIANT, "+ Team SP"),
    ("Estimation gap alone", "+ Estimation Gap", "No Effort"),
    ("Original design (pred effort)", "+ Pred Effort", "No Effort"),
    ("Both estimates vs none", PROPOSED_VARIANT, "No Effort"),
    ("Leakage: final SP vs prediction-time SP", LEAKAGE_VARIANT, PROPOSED_VARIANT),
]


def bootstrap_delta(y, p_base, p_new, seed=RANDOM_SEED, n_boot=N_BOOTSTRAP):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    deltas, tries = [], 0
    while len(deltas) < n_boot and tries < n_boot * 5:
        tries += 1
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(roc_auc_score(y[idx], p_new[idx]) - roc_auc_score(y[idx], p_base[idx]))
    point = roc_auc_score(y, p_new) - roc_auc_score(y, p_base)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"delta": round(float(point), 4), "ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4),
            "sig_pos": bool(lo > 0), "sig_neg": bool(hi < 0)}


def enough_classes(y):
    y = np.asarray(y)
    return int((y == 1).sum()) >= MIN_CLASS_COUNT and int((y == 0).sum()) >= MIN_CLASS_COUNT


def population(df, label):
    out = df[df[label].notna() & (df["eligible_for_risk_model"] == 1)].copy().reset_index(drop=True)
    out[label] = out[label].astype(int)
    return out


def add_stage1_predictions(fit_df, eval_df, effort_name, baselines, team_col):
    """Out-of-fold predicted effort for fit_df rows, full-fit prediction for eval_df rows."""
    fit_df, eval_df = fit_df.copy().reset_index(drop=True), eval_df.copy().reset_index(drop=True)
    fit_df["pred_effort"] = oof_effort_predictions(fit_df, effort_name, cat_baselines=baselines)
    pipe = build_effort_pipeline(effort_name, baselines).fit(fit_df, np.log1p(fit_df["story_points"].values))
    eval_df["pred_effort"] = predict_effort(pipe, eval_df)
    return add_estimate_features(fit_df, team_col), add_estimate_features(eval_df, team_col), pipe


def stage2_fit_predict(train_done, eval_done, baselines, variants, label):
    out = {}
    y = train_done[label].values
    for family in RISK_CANDIDATES:
        for variant, num in variants.items():
            pipe = build_risk_pipeline(family, cat_baselines=baselines, num_features=num)
            pipe.fit(train_done, y)
            out[(family, variant)] = (pipe, pipe.predict_proba(eval_done)[:, 1])
    return out


def logistic_coefs(pipe):
    names = pipe.named_steps["preprocessor"].get_feature_names_out()
    coefs = pipe.named_steps["model"].coef_.ravel()
    keep = ("estimation_gap", "log_sp_team", "log_sp_model", "pred_effort", "log_sp_final")
    return {n.split("__")[1]: round(float(c), 4) for n, c in zip(names, coefs) if n.split("__")[1] in keep}


def run_project(df, key, cfg):
    label, team_col = cfg["label"], cfg["team_col"]
    variants = make_risk_variants(cfg["base"])
    df = add_full_text(df)
    train = df[df["split"] == "train"].reset_index(drop=True)
    val = df[df["split"] == "val"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    trainval = pd.concat([train, val], ignore_index=True)
    baselines = most_common_baselines(train)
    res = {"project": key, "n_tickets": len(df), "reference_categories": baselines}

    # quick class-count check on test before spending time on training
    y_test_all = population(test, label)[label].values
    res["n_test"] = len(y_test_all)
    res["test_delay_rate"] = round(float(y_test_all.mean()), 3) if len(y_test_all) else None
    if not enough_classes(y_test_all):
        res["skipped"] = (f"test set has {int((y_test_all == 1).sum())} delayed / {int((y_test_all == 0).sum())} "
                          f"on-time tickets (rule: >= {MIN_CLASS_COUNT} of each)")
        return res

    # ---------------- Stage 1: select on validation ----------------
    val_mae = {}
    for name in EFFORT_CANDIDATES:
        pipe = build_effort_pipeline(name, baselines).fit(train, np.log1p(train["story_points"].values))
        val_mae[name] = eval_effort_metrics(val["story_points"].values, predict_effort(pipe, val))["MAE"]
    effort_name = min(val_mae, key=val_mae.get)
    res["effort_model"], res["effort_val_mae"] = effort_name, val_mae

    # ---------------- Stage 2: select family on validation ----------------
    tr_e, va_e, _ = add_stage1_predictions(train, val, effort_name, baselines, team_col)
    tr_done, va_done = population(tr_e, label), population(va_e, label)
    val_fits = stage2_fit_predict(tr_done, va_done, baselines, variants, label)
    y_val = va_done[label].values
    val_auc = {f"{fam} ({var})": (round(roc_auc_score(y_val, p), 4) if len(np.unique(y_val)) > 1 else None)
               for (fam, var), (_, p) in val_fits.items()}
    proposed_val = {fam: val_auc[f"{fam} ({PROPOSED_VARIANT})"] or 0 for fam in RISK_CANDIDATES}
    risk_family = max(proposed_val, key=proposed_val.get)
    res["risk_family"], res["risk_val_auc"] = risk_family, val_auc

    # ---------------- Test: refit on train+val, score once ----------------
    tv_e, te_e, effort_pipe = add_stage1_predictions(trainval, test, effort_name, baselines, team_col)
    dummy = Pipeline([("preprocessor", build_preprocessor(STAGE1_NUM_FEATURES, cat_baselines=baselines)),
                      ("model", DummyRegressor(strategy="median"))]).fit(trainval, np.log1p(trainval["story_points"]))
    res["effort_test"] = eval_effort_metrics(test["story_points"].values, predict_effort(effort_pipe, test))
    res["effort_test_dummy"] = eval_effort_metrics(test["story_points"].values, predict_effort(dummy, test))

    tv_done, te_done = population(tv_e, label), population(te_e, label)
    y_test = te_done[label].values
    res["team_sp_known_test_pct"] = round(100 * (1 - te_done["sp_missing_team"].mean()), 1)

    test_fits = stage2_fit_predict(tv_done, te_done, baselines, variants, label)
    res["risk_test"] = {f"{fam} ({var})": eval_risk_metrics(y_test, pipe.predict(te_done), p)
                        for (fam, var), (pipe, p) in test_fits.items()}
    res["comparisons"] = {}
    for fam in RISK_CANDIDATES:
        for name, new, base in COMPARISONS:
            res["comparisons"][f"{fam}|{name}"] = bootstrap_delta(y_test, test_fits[(fam, base)][1],
                                                                  test_fits[(fam, new)][1])

    # H3: tickets the team had NOT estimated at prediction time
    missing = te_done["sp_missing_team"].values == 1
    res["h3_n_unestimated"] = int(missing.sum())
    res["h3"] = {}
    if enough_classes(y_test[missing]):
        for fam in RISK_CANDIDATES:
            res["h3"][fam] = bootstrap_delta(y_test[missing], test_fits[(fam, "+ Team SP")][1][missing],
                                             test_fits[(fam, PROPOSED_VARIANT)][1][missing])

    res["logistic_coefs"] = {
        "gap_variant": logistic_coefs(test_fits[("Logistic Regression", "+ Estimation Gap")][0]),
        "both_variant": logistic_coefs(test_fits[("Logistic Regression", PROPOSED_VARIANT)][0]),
        "team_variant": logistic_coefs(test_fits[("Logistic Regression", "+ Team SP")][0]),
    }
    return res


def summarise(results):
    done = [r for r in results if "comparisons" in r]
    rows = []
    for scope in ["Selected"] + list(RISK_CANDIDATES):
        for name, new, base in COMPARISONS + [("H3 model estimate, unestimated tickets", PROPOSED_VARIANT, "+ Team SP")]:
            deltas, sig_pos, sig_neg = [], 0, 0
            for r in done:
                fam = r["risk_family"] if scope == "Selected" else scope
                c = r["h3"].get(fam) if name.startswith("H3") else r["comparisons"][f"{fam}|{name}"]
                if c is None:
                    continue
                deltas.append(c["delta"])
                sig_pos += c["sig_pos"]
                sig_neg += c["sig_neg"]
            try:
                p = wilcoxon(deltas).pvalue if len(deltas) >= 5 and any(d != 0 for d in deltas) else None
            except ValueError:
                p = None
            rows.append({"models": scope, "comparison": name, "variant_vs_baseline": f"{new} vs {base}",
                         "n_projects": len(deltas), "projects_improved": int(sum(d > 0 for d in deltas)),
                         "significant_improvement": int(sig_pos), "significant_worse": int(sig_neg),
                         "median_delta_auc": round(float(np.median(deltas)), 4) if deltas else None,
                         "mean_delta_auc": round(float(np.mean(deltas)), 4) if deltas else None,
                         "wilcoxon_p": round(float(p), 4) if p is not None else None})

    def sign_count(variant, feature, positive):
        vals = [r["logistic_coefs"][variant].get(feature) for r in done]
        vals = [v for v in vals if v is not None]
        return {"projects": len(vals), "in_hypothesis_direction": int(sum((v > 0) if positive else (v < 0) for v in vals)),
                "median": round(float(np.median(vals)), 4) if vals else None}

    # In the '+ Team SP + Model SP' model, b_team*log(team) + b_model*log(model)
    #   = (b_team + b_model)*log(team) - b_model*gap,  since log(model) = log(team) - gap.
    # So, holding the team's estimate fixed, the gap's effect is -b_model: b_model > 0 means
    # "the model reads the ticket as bigger than the team committed -> more delay" (the hypothesis).
    # The gap ALONE (without team SP) also carries ticket size, so its sign is not the hypothesis test.
    direction = {
        "team SP beta > 0 (bigger commitments -> more delay)": sign_count("team_variant", "log_sp_team", True),
        "model SP beta > 0 given team SP (= under-estimation -> more delay, size held fixed) [H2 direction]":
            sign_count("both_variant", "log_sp_model", True),
        "gap-alone beta < 0 (NOT size-controlled; mixes in ticket size - do not use as the H2 test)":
            sign_count("gap_variant", "estimation_gap", False),
    }
    return pd.DataFrame(rows), direction


def main():
    args = sys.argv[1:]
    label_name = "spillover"
    if "--label" in args:
        i = args.index("--label")
        label_name = args[i + 1]
        args = args[:i] + args[i + 2:]
    cfg = LABELS[label_name]
    data = pd.read_csv(cfg["dataset"])
    data[CAT_FEATURES] = data[CAT_FEATURES].fillna("None").astype(str)   # e.g. priority stored as "None"
    only = {k.upper() for k in args}
    projects = [p for p in data["project"].unique() if not only or p in only]
    print(f"Label: {label_name} | dataset: {cfg['dataset']} | projects: {len(projects)}")

    results, flat = [], []
    for key in projects:
        print(f"\n=== {key} ===")
        try:
            r = run_project(data[data["project"] == key].copy(), key, cfg)
        except Exception as exc:
            print(f"  !! failed: {exc}")
            continue
        results.append(r)
        if "comparisons" not in r:
            print(f"  skipped: {r['skipped']}")
            continue
        sel = r["risk_family"]
        auc = {v: r["risk_test"][f"{sel} ({v})"]["ROC-AUC"] for v in make_risk_variants([])}
        print(f"  Stage 1: {r['effort_model']} | test MAE {r['effort_test']['MAE']} (dummy {r['effort_test_dummy']['MAE']})")
        print(f"  Stage 2 (selected on val): {sel} | test n={r['n_test']} delay rate {r['test_delay_rate']} | "
              f"team SP known at prediction time: {r['team_sp_known_test_pct']}%")
        print("  test AUC: " + " | ".join(f"{v}: {a}" for v, a in auc.items()))
        for name, _, _ in COMPARISONS[:2]:
            c = r["comparisons"][f"{sel}|{name}"]
            print(f"  {name}: {c['delta']:+.4f} [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]")
        h3 = r["h3"].get(sel)
        print(f"  H3 (unestimated tickets, n={r['h3_n_unestimated']}): "
              + (f"{h3['delta']:+.4f} [{h3['ci_low']:+.4f}, {h3['ci_high']:+.4f}]" if h3 else "too few of each class"))
        row = {"project": key, "effort_model": r["effort_model"], "effort_test_mae": r["effort_test"]["MAE"],
               "effort_dummy_mae": r["effort_test_dummy"]["MAE"], "risk_family": sel, "n_test": r["n_test"],
               "test_delay_rate": r["test_delay_rate"], "team_sp_known_test_pct": r["team_sp_known_test_pct"],
               "lr_gap_beta": r["logistic_coefs"]["gap_variant"].get("estimation_gap"),
               "lr_model_sp_beta_given_team": r["logistic_coefs"]["both_variant"].get("log_sp_model"),
               "lr_team_sp_beta": r["logistic_coefs"]["team_variant"].get("log_sp_team")}
        row.update({f"AUC {v}": a for v, a in auc.items()})
        for name, _, _ in COMPARISONS:
            c = r["comparisons"][f"{sel}|{name}"]
            row[f"d[{name}]"] = c["delta"]
            row[f"CI[{name}]"] = f"[{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]"
        row["d[H3]"] = h3["delta"] if h3 else None
        flat.append(row)

    summary, direction = summarise(results)
    os.makedirs("reports", exist_ok=True)
    prefix = f"reports/mp_{label_name}"
    pd.DataFrame(flat).to_csv(f"{prefix}_results_per_project.csv", index=False)
    summary.to_csv(f"{prefix}_cross_project_summary.csv", index=False)
    skipped = {r["project"]: r["skipped"] for r in results if "skipped" in r}
    with open(f"{prefix}_results.json", "w") as f:
        json.dump({"label": label_name, "per_project": results, "direction": direction, "skipped": skipped},
                  f, indent=2, default=str)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    print("\n" + "=" * 120)
    print(f"CROSS-PROJECT SUMMARY - label: {label_name} (test sets; delta = AUC(variant) - AUC(baseline))")
    print("=" * 120)
    print(summary[summary["models"] == "Selected"].drop(columns=["models"]).to_string(index=False))
    print("\nCoefficient directions (logistic regression):")
    for k, v in direction.items():
        print(f"  {k}: {v}")
    if skipped:
        print("\nSkipped projects:")
        for k, v in skipped.items():
            print(f"  {k}: {v}")
    print(f"\nSaved {prefix}_results_per_project.csv, {prefix}_cross_project_summary.csv, {prefix}_results.json")
    print("(The summary CSV also has the same table for each model family.)")


if __name__ == "__main__":
    main()
