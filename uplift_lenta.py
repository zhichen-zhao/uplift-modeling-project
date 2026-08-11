"""
Uplift Modeling on the Lenta short-message-marketing RCT.

Lenta is a real randomized experiment from a Russian retailer: customers were
randomly assigned to receive an SMS (test) or not (control), and their purchase
response was tracked. Randomized treatment => a clean, unconfounded setting for
uplift modeling.

Methodology (the key discipline here):
  The data is split 80/20 ONCE at the very start.
    - development set (80%): used for BOTH out-of-fold model comparison AND
      refitting the final model.
    - held-out set (20%): touched ONLY at deployment/validation. The model never
      sees these customers during training, so scoring + validating on them is an
      honest test of real-world targeting performance.

Pipeline:
  1. Load Lenta, handle missing values, encode gender, binarize treatment.
  2. Split 80/20 into development / held-out sets.
  3. On development: out-of-fold CATE for S/T/X/DR learners; compare with Qini.
  4. Refit the best learner on all development data.
  5. Deploy on the held-out set: score, select the top 20% by predicted uplift,
     and VALIDATE with the held-out set's real T and Y.

Run (Colab or a networked machine):
  pip install scikit-uplift econml causalml scikit-learn
  python uplift_lenta.py
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import (HistGradientBoostingRegressor,
                              HistGradientBoostingClassifier)
from econml.metalearners import SLearner, TLearner, XLearner
from econml.dr import DRLearner
from causalml.metrics import plot_qini, qini_score

from sklift.datasets import fetch_lenta


# ------------------------------------------------------------
# 1) Load and prepare the Lenta RCT
# ------------------------------------------------------------
def load_data():
    data, target, treatment = fetch_lenta(return_X_y_t=True)

    # (1) Drop feature columns with >50% missing; they carry too little signal.
    missing_frac = data.isnull().mean()
    cols_keep = missing_frac[missing_frac <= 0.5].index
    df_feat = data[cols_keep].copy()

    # (2) gender is the only string feature. Missing gender = "undetermined",
    #     so fold missing into the existing "Не определен" category, keep as str.
    df_feat["gender"] = df_feat["gender"].fillna("Не определен").astype(str)

    # (3) Remaining (numeric) missing means "no such purchase activity" -> 0.
    num_cols = [c for c in df_feat.columns if c != "gender"]
    df_feat[num_cols] = df_feat[num_cols].fillna(0)

    # (4) One-hot encode gender; stack with the numeric features.
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat = enc.fit_transform(df_feat[["gender"]])
    X_num = df_feat[num_cols].to_numpy(dtype=float)
    X = np.hstack([X_num, X_cat]).astype(float)

    # (5) Treatment: SMS sent (test) = 1, none (control) = 0.  Outcome: response.
    T = (treatment == "test").astype(int).values
    Y = target.astype(float).values
    return X, T, Y


# ------------------------------------------------------------
# 2) Fast base models (HistGradientBoosting: 10-100x faster than
#    plain GradientBoosting, which is far too slow at this scale)
# ------------------------------------------------------------
def gbr():
    return HistGradientBoostingRegressor(random_state=0)


def gbc():
    return HistGradientBoostingClassifier(random_state=0)


def build_learners():
    return {
        "S-learner":  lambda: SLearner(overall_model=gbr()),
        "T-learner":  lambda: TLearner(models=gbr()),
        # pass an explicit propensity model so it converges cleanly
        "X-learner":  lambda: XLearner(models=gbr(), propensity_model=gbc()),
        "DR-learner": lambda: DRLearner(model_propensity=gbc(),
                                        model_regression=gbr(),
                                        model_final=gbr()),
    }


# ------------------------------------------------------------
# 3) Out-of-fold CATE (each row scored by a model that did NOT train on it)
# ------------------------------------------------------------
def oof_cate(model_ctor, X, T, Y, n_splits=5, seed=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cate = np.zeros(len(Y))
    for tr, te in kf.split(X):
        m = model_ctor()
        m.fit(Y[tr], T[tr], X=X[tr])
        cate[te] = m.effect(X[te])
    return cate


# ------------------------------------------------------------
# Helper: actual uplift within a group, from real T and Y
#   = (response rate among treated) - (response rate among control)
# ------------------------------------------------------------
def actual_uplift(mask, T, Y):
    treated = (T == 1) & mask
    control = (T == 0) & mask
    return Y[treated].mean() - Y[control].mean()


# ------------------------------------------------------------
# 4) Main
# ------------------------------------------------------------
def main():
    X, T, Y = load_data()
    print(f"Full data: X {X.shape} | T mean {T.mean():.3f} | Y mean {Y.mean():.3f}")

    # -- Split ONCE, up front. dev = modeling; hold = deployment only. --
    X_dev, X_hold, T_dev, T_hold, Y_dev, Y_hold = train_test_split(
        X, T, Y, test_size=0.2, random_state=0
    )
    print(f"Development set: {X_dev.shape[0]} | Held-out set: {X_hold.shape[0]}")

    # -- Compare learners with out-of-fold CATE, ON THE DEVELOPMENT SET ONLY --
    learners = build_learners()
    cate_results = {}
    for name, ctor in learners.items():
        print(f"Fitting {name} ...")
        cate_results[name] = oof_cate(ctor, X_dev, T_dev, Y_dev)

    qini_df = pd.DataFrame({
        "y": Y_dev,
        "w": T_dev,
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

    # -- Refit the best learner on ALL development data --
    learner_key = best_name.replace("_", "-")
    final_model = build_learners()[learner_key]()
    final_model.fit(Y_dev, T_dev, X=X_dev)

    # -- Deploy on the HELD-OUT set (never seen in training) --
    cate_hold = final_model.effect(X_hold)
    target_mask = cate_hold >= np.percentile(cate_hold, 80)   # top 20% by predicted uplift

    print("\n--- Deployment on held-out customers ---")
    print(f"Held-out customers scored : {len(cate_hold)}")
    print(f"Selected to target (top 20%): {target_mask.sum()}")

    # -- VALIDATE with the held-out set's real T and Y --
    up_sel  = actual_uplift(target_mask,  T_hold, Y_hold)
    up_rest = actual_uplift(~target_mask, T_hold, Y_hold)
    up_all  = actual_uplift(np.ones_like(target_mask, dtype=bool), T_hold, Y_hold)
    print("\n--- Actual uplift on held-out set (real T, Y) ---")
    print(f"Selected top 20% : {up_sel:.4f}")
    print(f"The rest 80%     : {up_rest:.4f}")
    print(f"Whole held-out   : {up_all:.4f}")
    print(f"Targeting lift   : {up_sel - up_all:+.4f} above population average")

    return final_model


if __name__ == "__main__":
    main()
