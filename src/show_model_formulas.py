"""
show_model_formulas.py
----------------------
Prints the ACTUAL learned formulas of the models used in this project, so they
can be shown in the thesis / viva.

  1. Risk-label regression      log(1+cycle_time) = b0 + b1 * story_points
  2. Effort model (Ridge)       log(1+SP) = b0 + b1*x1 + b2*x2 + ...
  3. Effort model (Ridge+TF-IDF) the same, plus one beta per word/phrase
  4. Risk model (Logistic Reg.) P(at risk) = 1 / (1 + e^-(b0 + b1*x1 + ...))
  5. A hand-calculation for one ticket proving the printed formula IS what
     model.predict_proba() computes.
  6. LightGBM                    no single equation: sum of many decision trees

Run from the project root:   python src/show_model_formulas.py
"""
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.model_selection import KFold
from sklearn.base import clone

DATA = os.environ.get("DATA_DIR", "data/processed")
pd.set_option("display.width", 200)

CAT = ["issuetype", "priority"]
S1_NUM = ["num_components", "num_labels", "summary_len", "description_len",
          "reporter_prev_issue_count", "priority_ordinal",
          "has_description", "has_components", "has_labels"]
S2_NUM = S1_NUM + ["early_status_changes", "early_assignee_changes", "early_priority_changes",
                   "early_comment_count", "had_early_status_change",
                   "first_status_change_days_imputed"]


def line(t=""):
    print("\n" + "=" * 90 + ("\n" + t + "\n" + "=" * 90 if t else ""))


def coef_table(pipe):
    names = pipe.named_steps["preprocessor"].get_feature_names_out()
    model = pipe.named_steps["model"]
    beta = model.coef_.ravel()
    return pd.DataFrame({"feature": names, "beta": beta}), float(np.ravel(model.intercept_)[0])


def unscale(pipe, tab):
    """Convert betas of standardised numeric features back to raw units:
       beta_raw = beta / std ; intercept_raw = b0 - sum(beta * mean / std)."""
    scaler = pipe.named_steps["preprocessor"].named_transformers_["num"]
    nums = pipe.named_steps["preprocessor"].transformers_[1][2]
    raw = tab.copy()
    raw["beta_raw_units"] = raw["beta"]
    shift = 0.0
    for i, f in enumerate(nums):
        m = raw["feature"] == f"num__{f}"
        b = raw.loc[m, "beta"].values[0]
        raw.loc[m, "beta_raw_units"] = b / scaler.scale_[i]
        shift += b * scaler.mean_[i] / scaler.scale_[i]
    return raw, shift


df = pd.read_csv(f"{DATA}/xd_modeling_dataset.csv")
df["full_text"] = df["summary"].fillna("") + " " + df["description"].fillna("")
train = df[df["split"] == "train"].reset_index(drop=True)

# ---------------------------------------------------------------------------
line("1. RISK LABEL: log-linear regression (build_labels_and_splits.py, lines 69-79)")
done = pd.read_csv(f"{DATA}/xd_done_with_risk_candidates.csv")
td = done[done["split"] == "train"]
lr = LinearRegression().fit(td[["story_points"]], np.log1p(td["cycle_time_days"]))
b0, b1 = lr.intercept_, lr.coef_[0]
print(f"log(1 + cycle_time_days) = {b0:.4f} + {b1:.4f} * story_points")
print(f"=> expected cycle time   = exp({b0:.4f} + {b1:.4f}*SP) - 1")
for sp in [1, 3, 8]:
    print(f"   SP={sp}: expected {np.expm1(b0 + b1*sp):.1f} days")
print("residual = actual log-time - expected log-time; top 25% of train residuals -> at risk")

# ---------------------------------------------------------------------------
line("2. EFFORT MODEL: Ridge, tabular only (target = log(1 + story_points))")
prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
                          ("num", StandardScaler(), S1_NUM)])
ridge = Pipeline([("preprocessor", prep), ("model", Ridge(alpha=10.0))])
ridge.fit(train, np.log1p(train["story_points"]))
tab, b0 = coef_table(ridge)
raw, shift = unscale(ridge, tab)
print("Ridge minimises:  SUM_i (y_i - (b0 + b.x_i))^2  +  alpha * SUM_j b_j^2   (alpha = 10)")
print(f"\nIntercept b0 = {b0:.4f}   (numeric features are z-scores: (x - mean)/std)")
print(raw.round(5).to_string(index=False))
print(f"\nIn raw units the intercept becomes {b0 - shift:.4f}")
eq = " + ".join(f"({r.beta_raw_units:+.5f} * {r.feature.split('__')[1]})" for r in raw.itertuples())
print(f"\nlog(1+SP) = {b0 - shift:.4f} + {eq}")
print("Story Points = exp(that) - 1")

# ---------------------------------------------------------------------------
line("3. EFFORT MODEL: Ridge + TF-IDF (validation-best Stage-1 model)")
prep_t = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
                            ("num", StandardScaler(), S1_NUM),
                            ("text", TfidfVectorizer(max_features=300, stop_words="english",
                                                     ngram_range=(1, 2), sublinear_tf=True), "full_text")])
ridge_t = Pipeline([("preprocessor", prep_t), ("model", Ridge(alpha=10.0))])
ridge_t.fit(train, np.log1p(train["story_points"]))
tab_t, b0t = coef_table(ridge_t)
print(f"Same formula, now with {len(tab_t)} betas ({(tab_t.feature.str.startswith('text__')).sum()} are words/phrases).")
words = tab_t[tab_t.feature.str.startswith("text__")].sort_values("beta")
print("\nWords that push the estimate UP (largest +beta):")
print(words.tail(12)[::-1].round(4).to_string(index=False))
print("\nWords that push the estimate DOWN (most negative beta):")
print(words.head(12).round(4).to_string(index=False))

# ---------------------------------------------------------------------------
line("4. RISK MODEL: Logistic Regression + predicted effort (validation-best Stage-2 model)")
kf = KFold(5, shuffle=True, random_state=42)
oof = np.zeros(len(train))
for a, b in kf.split(train):
    p = clone(ridge_t).fit(train.iloc[a], np.log1p(train.iloc[a]["story_points"]))
    oof[b] = np.expm1(p.predict(train.iloc[b])).clip(min=0)
train["pred_effort"] = oof
rd = train[(train["risk_25pct"].notna()) & (train["eligible_for_risk_model"] == 1)].reset_index(drop=True)
y = rd["risk_25pct"].astype(int)
prep_r = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
                            ("num", StandardScaler(), S2_NUM + ["pred_effort"])])
logit = Pipeline([("preprocessor", prep_r),
                  ("model", LogisticRegression(class_weight="balanced", max_iter=1000))])
logit.fit(rd, y)
tab_r, b0r = coef_table(logit)
tab_r["odds_ratio_per_1sd"] = np.exp(tab_r["beta"])
print("Logistic regression minimises the log-loss (cross-entropy):")
print("   L = - SUM_i w_i [ y_i*log(p_i) + (1-y_i)*log(1-p_i) ]  +  (1/C)*0.5*SUM_j b_j^2   (C = 1)")
print("   where  z_i = b0 + b1*x_i1 + ... + bk*x_ik   and   p_i = 1 / (1 + e^(-z_i))")
print("   w_i = class weight ('balanced' = n / (2 * n_class)) so at-risk rows count more")
print(f"\nIntercept b0 = {b0r:.4f}")
print(tab_r.sort_values("beta", key=abs, ascending=False).round(4).to_string(index=False))

# ---------------------------------------------------------------------------
line("5. PROOF: hand-calculate one ticket and compare with model.predict_proba()")
x = rd.iloc[[0]]
X = logit.named_steps["preprocessor"].transform(x)
X = X.toarray() if hasattr(X, "toarray") else X
z = b0r + float((X @ tab_r["beta"].values)[0])
print(f"Ticket {x['key'].values[0]}:")
for name, xv, bv in zip(tab_r["feature"], X.ravel(), tab_r["beta"]):
    if abs(xv) > 1e-12:
        print(f"   {name:45s} x = {xv:+.4f}   beta = {bv:+.4f}   beta*x = {xv*bv:+.4f}")
print(f"   z = b0 + SUM(beta*x) = {z:.4f}")
print(f"   p = 1/(1+e^-z)       = {1/(1+np.exp(-z)):.6f}")
print(f"   model.predict_proba  = {logit.predict_proba(x)[0,1]:.6f}   <- identical")

# ---------------------------------------------------------------------------
line("6. LightGBM: no single equation - prediction = SUM of tree outputs")
try:
    from lightgbm import LGBMClassifier
    prep_l = clone(prep_r)
    lgb = Pipeline([("preprocessor", prep_l),
                    ("model", LGBMClassifier(n_estimators=100, is_unbalance=True,
                                             random_state=42, verbose=-1))]).fit(rd, y)
    booster = lgb.named_steps["model"].booster_
    names = list(prep_l.get_feature_names_out())
    tdf = booster.trees_to_dataframe()
    print(f"{booster.num_trees()} trees, {len(tdf)} nodes in total.")
    print("raw score F(x) = tree_1(x) + tree_2(x) + ... + tree_100(x);   P(at risk) = 1/(1+e^-F(x))")
    print("Each tree is fitted to the gradient of the log-loss of the trees before it.\n")
    t0 = tdf[tdf.tree_index == 0].copy()
    t0["split_feature"] = t0["split_feature"].map(
        lambda s: names[int(s.split("_")[1])] if isinstance(s, str) and s.startswith("Column_") else s)
    print("Tree #1 (first 15 nodes):")
    print(t0[["node_index", "split_feature", "threshold", "left_child", "right_child", "value"]]
          .head(15).to_string(index=False))
except ImportError:
    print("lightgbm not installed here - run this script in the project venv to see the trees.")
