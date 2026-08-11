# Uplift Modeling for Treatment Targeting

Comparing four causal **meta-learners** (S-, T-, X-, and DR-learner) for
estimating individual treatment effects (CATE) and deciding **whom to target**,
across three datasets of increasing realism: a synthetic benchmark, and two
real marketing randomized experiments (Hillstrom and Lenta).

The central question of uplift modeling is not "who will respond?" but **"whose
response is *caused* by the treatment?"** — the people worth spending budget on.
This project estimates that per-person causal effect, ranks people by it, and
validates the ranking honestly on held-out data.

## What this project shows

Running the same pipeline on three datasets surfaces one consistent, practical
lesson:

> **Model complexity must match the problem, not exceed it.** The simplest
> learners (S, T) are the most robust; the more elaborate ones (X, DR) frequently
> underperform and are sensitive to sample size. Complexity that isn't needed
> becomes a source of noise, not a source of gain.

| Dataset | Samples | Signal | Best learner | Key observation |
|---|---|---|---|---|
| Synthetic | 5k | strong | **S-learner** | simplest learner wins |
| Hillstrom (email RCT) | 42k | weak | **T-learner** (≈ random) | real email-marketing signal is faint |
| Lenta (SMS RCT) | 687k | moderate | **T-learner** | T robust; **X-learner worst**; targeting doubles real uplift |

*(Exact numbers depend on the run; the pattern — simple beats complex — is what
holds up across all three.)*

## Method

Each script follows the same disciplined pipeline:

1. **Load & prepare** the data (for real data: binarize treatment, encode
   categoricals, handle missing values).
2. **Split 80/20 once, up front.** The development set (80%) is used for model
   comparison and refitting; the held-out set (20%) is touched *only* at
   deployment. This keeps the deployment test honest — the final model scores
   people it never trained on.
3. **Compare learners with out-of-fold CATE** on the development set: every
   person's effect is predicted by a model that did *not* train on them.
4. **Evaluate rankings with the Qini curve and Qini score**, which use only the
   observed outcome `Y` and treatment `T` — exactly what's available in
   deployment (no ground-truth effect needed).
5. **Deploy**: refit the best learner on all development data, score the
   held-out set, select the top 20% by predicted uplift, and — for the real
   datasets — **validate** with the held-out set's real `T` and `Y` (compare the
   actual response lift of the targeted group vs. the rest).

## The four meta-learners

Each estimates the CATE, `τ(x) = E[Y(1) − Y(0) | X = x]`, a different way:

- **S-learner** — one model with treatment as an input feature; `τ = μ(x,1) − μ(x,0)`.
- **T-learner** — two separate models, one per arm; `τ = μ₁(x) − μ₀(x)`.
- **X-learner** — T-learner plus a cross-imputation step; designed for
  imbalanced arms.
- **DR-learner** — doubly-robust: combines outcome regressions with a propensity
  model, so the estimate stays valid if *either* is correct. This is the
  AIPW / influence-function construction from semiparametric causal inference.

Complexity increases S < T < X < DR. Across all three datasets, that extra
complexity did **not** pay off.

## The datasets

- **Synthetic** (`uplift_synthetic.py`) — a controlled benchmark with a strong,
  known effect. The "easy mode" that confirms the pipeline works when signal is
  abundant. The true effect is deliberately discarded; evaluation uses only
  observed `Y` and `T`, as in reality.
- **Hillstrom** (`uplift_hillstrom.py`) — the MineThatData email-marketing RCT.
  `T` = received the Men's email vs. no email; `Y` = visited the site within two
  weeks. A genuine but **weak-signal** case: learners land close to random.
- **Lenta** (`uplift_lenta.py`) — a large SMS-marketing RCT from the Lenta /
  Microsoft *BigTarget* hackathon. `T` = received a marketing contact vs. not;
  `Y` = customer response (binary, ~11%); `X` = ~190 anonymized features, mostly
  historical grocery-purchase statistics per time window and product group, plus
  gender / age / store type. The **moderate-signal** case where targeting works:
  the top-20% selected by the model showed roughly double the real uplift of the
  population average, validated on a fully held-out set.

## Why randomized data matters

All three treatments are **randomized**, so the treated and control groups are
statistically balanced and the difference in outcomes can be attributed cleanly
to the treatment (no confounding). This makes them honest settings for uplift.
On observational data the same estimators can be biased, which is exactly where
doubly-robust methods and stronger assumptions (unconfoundedness) come in.

## Repository structure

```
uplift-modeling-project/
├── README.md
├── requirements.txt
├── uplift_synthetic.py     # synthetic benchmark (strong signal)
├── uplift_hillstrom.py     # Hillstrom email RCT (weak signal)
└── uplift_lenta.py         # Lenta SMS RCT (moderate signal, targeting validated)
```

## Running

```bash
pip install -r requirements.txt
python uplift_synthetic.py     # runs in seconds
python uplift_hillstrom.py     # downloads Hillstrom via scikit-uplift
python uplift_lenta.py         # large; see note below
```

**Note on Lenta.** It has ~687k rows and ~190 features. The scripts use
`HistGradientBoosting` (far faster than plain `GradientBoosting`) to keep this
tractable, but a full run still takes several minutes — the DR-learner is the
slowest. On a modest machine, subsample to ~150k rows first (the "simple beats
complex" pattern is already clear there).
