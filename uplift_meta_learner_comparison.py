"""
Uplift Modeling: comparing four causal meta-learners for treatment targeting.

Pipeline:
  1. Generate synthetic continuous-outcome data (true effect ignored, as in reality).
  2. Estimate CATE out-of-fold (K-fold CV) with S / T / X / DR learners.
  3. Evaluate rankings with the Qini curve and Qini score (uses only Y and T).

Run:
  pip install -r requirements.txt
  python uplift_meta_learner_comparison.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from econml.metalearners import SLearner, TLearner, XLearner
from econml.dr import DRLearner
from causalml.dataset import synthetic_data
from causalml.metrics import plot_qini, qini_score


# ------------------------------------------------------------
# 1) Generate data (we deliberately do NOT use the true effect)
# ------------------------------------------------------------
def load_data(n=5000, p=5, sigma=0.5, seed=2026):
    np.random.seed(seed)
    Y, X, T, tau_true, b, e = synthetic_data(mode=1, n=n, p=p, sigma=sigma)
    return Y.astype(float), X, T.astype(int)   # tau_true intentionally dropped


# ------------------------------------------------------------
# 2) Out-of-fold CATE via K-fold cross-validation
#    (each row scored by a model that did NOT train on it)
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
# 3) The four meta-learners
# ------------------------------------------------------------
def gbr():
    return GradientBoostingRegressor(random_state=0)


def gbc():
    return GradientBoostingClassifier(random_state=0)


def build_learners():
    return {
        "S-learner":  lambda: SLearner(overall_model=gbr()),
        "T-learner":  lambda: TLearner(models=gbr()),
        "X-learner":  lambda: XLearner(models=gbr()),
        "DR-learner": lambda: DRLearner(model_propensity=gbc(),
                                        model_regression=gbr(),
                                        model_final=gbr()),
    }


# ------------------------------------------------------------
# 4) Main
# ------------------------------------------------------------
def main():
    Y, X, T = load_data()
    print(f"X: {X.shape} | T mean: {T.mean():.3f} | Y mean: {Y.mean():.3f}")

    learners = build_learners()
    cate_results = {}
    for name, ctor in learners.items():
        print(f"Fitting {name} ...")
        cate_results[name] = oof_cate(ctor, X, T, Y)

    # Assemble a frame: outcome, treatment, and one CATE column per learner
    qini_df = pd.DataFrame({
        "y": Y,
        "w": T,
        "S_learner":  cate_results["S-learner"],
        "T_learner":  cate_results["T-learner"],
        "X_learner":  cate_results["X-learner"],
        "DR_learner": cate_results["DR-learner"],
    })

    # Qini curve (evaluation uses only observed Y and T)
    plot_qini(qini_df, outcome_col="y", treatment_col="w")

    # Qini scores (higher = better ranking)
    scores = qini_score(qini_df, outcome_col="y", treatment_col="w")
    print("\nQini score (higher is better):")
    print(scores)
    best_name = scores.drop("Random", errors="ignore").idxmax()
    print(f"\nBest learner: {best_name}")

    # --------------------------------------------------------
    # 5) Deploy: refit the best learner on ALL data, then use it
    #    to select whom to treat among NEW, unlabeled people.
    #    (CV models above were for evaluation only and are discarded;
    #     the final model is trained on the full dataset.)
    # --------------------------------------------------------
    learner_key = best_name.replace("_", "-")   # "S_learner" -> "S-learner"
    final_model = learners[learner_key]()
    final_model.fit(Y, T, X=X)

    # Stand-in for future arrivals: here we reuse X as an example.
    # In practice, replace X_new with the new people's features.
    X_new = X
    cate_new = final_model.effect(X_new)

    # Select the top 20% by predicted uplift — the people to target.
    threshold = np.percentile(cate_new, 80)
    target_mask = cate_new >= threshold

    print("\n--- Targeting on new data ---")
    print(f"New people scored: {len(cate_new)}")
    print(f"Selected to treat (top 20%): {target_mask.sum()}")
    print(f"Avg predicted uplift — targeted : {cate_new[target_mask].mean():.4f}")
    print(f"Avg predicted uplift — the rest : {cate_new[~target_mask].mean():.4f}")

    # `np.where(target_mask)[0]` gives the row indices of the people to treat.
    return final_model


if __name__ == "__main__":
    main()
