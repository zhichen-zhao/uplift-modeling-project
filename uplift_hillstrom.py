"""
Uplift Modeling on the Hillstrom MineThatData email-marketing RCT.

Hillstrom is a genuine randomized experiment: 64,000 customers were randomly
assigned to receive a Men's email, a Women's email, or no email, and their
website visits were tracked for two weeks. Randomized treatment => a clean,
unconfounded setting for uplift modeling.

This is the "weak-signal" case study: email-marketing uplift is genuinely faint
(most people behave about the same whether or not they get an email), so the
learners end up close to random. That is a real and instructive result, not a
bug -- it shows that a method that shines on synthetic data can be near-useless
when the real-world signal is weak.

Methodology (shared across all three scripts):
  Split 80/20 once, up front. dev (80%) = modeling; hold (20%) = deployment only.

Pipeline:
  1. Load Hillstrom; binarize treatment (Men's E-Mail = 1 vs No E-Mail = 0),
     one-hot encode categorical features, pick a binary outcome (visit).
  2. Split 80/20 into development / held-out.
  3. On development: out-of-fold CATE for S/T/X/DR; compare with Qini.
  4. Refit the best learner on all development data.
  5. Deploy on the held-out set: select top 20%, validate with real T and Y.

Run (Colab or a networked machine):
  pip install scikit-uplift econml causalml scikit-learn
  python uplift_hillstrom.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from econml.metalearners import SLearner, TLearner, XLearner
from econml.dr import DRLearner
from causalml.metrics import plot_qini, qini_score

from sklift.datasets import fetch_hillstrom


# ------------------------------------------------------------
# 1) Load and prepare the RCT
#    Real data needs three things synthetic data did not:
#      (a) treatment binarized to {0,1}
#      (b) categorical (string) features encoded to numbers
#      (c) a chosen outcome column
# ------------------------------------------------------------
def load_data():
    data, target, treatment = fetch_hillstrom(target_col="visit", return_X_y_t=True)
    df = data.copy()
    df["treatment"] = treatment
    df["visit"] = target.astype(float)

    # (a) Keep the Men's-email arm vs the control arm for a clean two-arm experiment.
    df = df[df["treatment"].isin(["Mens E-Mail", "No E-Mail"])].copy()
    T = (df["treatment"] == "Mens E-Mail").astype(int).values

    # (b) One-hot encode the string columns; keep the numeric ones.
    cat_cols = ["history_segment", "zip_code", "channel"]
    num_cols = ["recency", "history", "mens", "womens", "newbie"]
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat = enc.fit_transform(df[cat_cols])
    X_num = df[num_cols].to_numpy(dtype=float)
    X = np.hstack([X_num, X_cat]).astype(float)

    # (c) Outcome
    Y = df["visit"].to_numpy(dtype=float)
    return X, T, Y


# ------------------------------------------------------------
# 2) Base models and the four meta-learners
# ------------------------------------------------------------
def gbr():
    return GradientBoostingRegressor(random_state=0)


def gbc():
    return GradientBoostingClassifier(random_state=0)


def build_learners():
    return {
        "S-learner":  lambda: SLearner(overall_model=gbr()),
        "T-learner":  lambda: TLearner(models=gbr()),
        "X-learner":  lambda: XLearner(models=gbr(), propensity_model=gbc()),
        "DR-learner": lambda: DRLearner(model_propensity=gbc(),
                                        model_regression=gbr(),
                                        model_final=gbr()),
    }


# ------------------------------------------------------------
# 3) Out-of-fold CATE
# ------------------------------------------------------------
def oof_cate(model_ctor, X, T, Y, n_splits=5, seed=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cate = np.zeros(len(Y))
    for tr, te in kf.split(X):
        m = model_ctor()
        m.fit(Y[tr], T[tr], X=X[tr])
        cate[te] = m.effect(X[te])
    return cate


# actual uplift within a group, from real T and Y
def actual_uplift(mask, T, Y):
    treated = (T == 1) & mask
    control = (T == 0) & mask
    return Y[treated].mean() - Y[control].mean()


# ------------------------------------------------------------
# 4) Main
# ------------------------------------------------------------
def main():
    X, T, Y = load_data()
    print(f"Data: X {X.shape} | T mean {T.mean():.3f} | Y mean {Y.mean():.3f}")

    X_dev, X_hold, T_dev, T_hold, Y_dev, Y_hold = train_test_split(
        X, T, Y, test_size=0.2, random_state=0
    )
    print(f"Development set: {len(Y_dev)} | Held-out set: {len(Y_hold)}")

    learners = build_learners()
    cate_results = {}
    for name, ctor in learners.items():
        print(f"Fitting {name} ...")
        cate_results[name] = oof_cate(ctor, X_dev, T_dev, Y_dev)

    qini_df = pd.DataFrame({
        "y": Y_dev, "w": T_dev,
        "S_learner":  cate_results["S-learner"],
        "T_learner":  cate_results["T-learner"],
        "X_learner":  cate_results["X-learner"],
        "DR_learner": cate_results["DR-learner"],
    })
    plot_qini(qini_df, outcome_col="y", treatment_col="w")
    scores = qini_score(qini_df, outcome_col="y", treatment_col="w")
    print("\nQini score (higher is better):")
    print(scores)
    best_name = scores.drop("Random", errors="ignore").idxmax()
    print(f"Best learner: {best_name}")

    learner_key = best_name.replace("_", "-")
    final_model = build_learners()[learner_key]()
    final_model.fit(Y_dev, T_dev, X=X_dev)

    cate_hold = final_model.effect(X_hold)
    target_mask = cate_hold >= np.percentile(cate_hold, 80)

    print("\n--- Deployment on held-out customers ---")
    print(f"Held-out customers scored : {len(cate_hold)}")
    print(f"Selected to target (top 20%): {target_mask.sum()}")

    up_sel  = actual_uplift(target_mask,  T_hold, Y_hold)
    up_rest = actual_uplift(~target_mask, T_hold, Y_hold)
    up_all  = actual_uplift(np.ones_like(target_mask, dtype=bool), T_hold, Y_hold)
    print("\n--- Actual uplift on held-out set (real T, Y) ---")
    print(f"Selected top 20% : {up_sel:.4f}")
    print(f"The rest 80%     : {up_rest:.4f}")
    print(f"Whole held-out   : {up_all:.4f}")

    return final_model


if __name__ == "__main__":
    main()
