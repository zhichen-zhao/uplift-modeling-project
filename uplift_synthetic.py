"""
Uplift Modeling on synthetic data: comparing four causal meta-learners.

This is the controlled-experiment baseline for the project. The synthetic
generator produces a strong, known treatment-effect signal, so it is the "easy
mode" that shows the pipeline works when signal is abundant. (The two real-data
scripts -- Hillstrom and Lenta -- then show what happens when signal is weak or
moderate.)

Note: causalml's synthetic_data (mode=1) uses a covariate-dependent propensity
e(x), so this is an OBSERVATIONAL dataset (treatment correlated with features),
unlike the randomized Hillstrom and Lenta RCTs. Unbiasedness here rests on
unconfoundedness; the DR-learner's propensity model is what handles it.

We deliberately DISCARD the true effect and evaluate only with observed Y and T,
exactly as we would have to in the real world.

Methodology (shared across all three scripts):
  The data is split 80/20 once, up front.
    - development set (80%): out-of-fold model comparison + refitting the final model
    - held-out set (20%): touched only at deployment -- an honest test of targeting

Pipeline:
  1. Generate synthetic data (true effect ignored).
  2. Split 80/20 into development / held-out.
  3. On development: out-of-fold CATE for S/T/X/DR; compare with Qini.
  4. Refit the best learner on all development data.
  5. Deploy on the held-out set: select the top 20% by predicted uplift.

Run:
  pip install -r requirements.txt
  python uplift_synthetic.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
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
# 4) Main
# ------------------------------------------------------------
def main():
    Y, X, T = load_data()
    print(f"Data: X {X.shape} | T mean {T.mean():.3f} | Y mean {Y.mean():.3f}")

    # Split once, up front. dev = modeling; hold = deployment only.
    X_dev, X_hold, T_dev, T_hold, Y_dev, Y_hold = train_test_split(
        X, T, Y, test_size=0.2, random_state=0
    )
    print(f"Development set: {len(Y_dev)} | Held-out set: {len(Y_hold)}")

    # Compare learners with out-of-fold CATE, on the development set only
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

    # Refit the best learner on all development data
    learner_key = best_name.replace("_", "-")
    final_model = build_learners()[learner_key]()
    final_model.fit(Y_dev, T_dev, X=X_dev)

    # Deploy on the held-out set (never seen in training)
    cate_hold = final_model.effect(X_hold)
    target_mask = cate_hold >= np.percentile(cate_hold, 80)   # top 20% by predicted uplift

    print("\n--- Deployment on held-out data ---")
    print(f"Held-out units scored     : {len(cate_hold)}")
    print(f"Selected to treat (top 20%): {target_mask.sum()}")
    print(f"Avg predicted uplift, targeted : {cate_hold[target_mask].mean():.4f}")
    print(f"Avg predicted uplift, the rest : {cate_hold[~target_mask].mean():.4f}")

    return final_model


if __name__ == "__main__":
    main()
