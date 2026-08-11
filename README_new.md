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

| Dataset | Samples | Signal | Best learner (Qini) | Key observation |
|---|---|---|---|---|
| Synthetic | 5k | strong | **S-learner (0.15)** | simplest learner wins; X/DR no better |
| Hillstrom (email RCT) | 42k | weak | **T-learner (0.02)** | all learners ≈ random; email-marketing signal is faint |
| Lenta (SMS RCT) | 687k | moderate | **T-learner (0.16)** | T robust; **X-learner worst (−0.09)**; targeting doubles real uplift |

*(Exact numbers depend on the run and carry noise; the pattern — simple beats
complex, across all three datasets — is what holds up.)*

## Results

The synthetic and Hillstrom studies go as far as **modeling and evaluation**
(compare learners, read the Qini). Lenta goes one step further into
**deployment**: refit the winner, score a held-out set the model never saw, pick
the top 20%, and check the *real* uplift of those selected people.

### Synthetic (strong signal) — modeling + evaluation

| Learner | Qini score |
|---|---|
| **S-learner** | **0.154** |
| T-learner | 0.147 |
| X-learner | 0.130 |
| DR-learner | 0.129 |

<img src="images/synthetic_qini.png" width="55%">

With a strong, clean effect, all four rank well above random and the **simplest
learner (S) wins**. The extra machinery of X-/DR-learners buys nothing here —
their advantages (group imbalance, misspecified models) aren't stressed by this
data.

### Hillstrom email RCT (weak signal) — modeling + evaluation

| Learner | Qini score |
|---|---|
| **T-learner** | **0.020** |
| S-learner | 0.013 |
| X-learner | 0.011 |
| DR-learner | −0.004 |

<img src="images/hillstrom_qini.png" width="55%">

*All four curves hug the random diagonal.* Email-marketing uplift is genuinely
faint — most customers behave about the same whether or not they get an email —
so no learner ranks much better than random, and the differences between them are
within noise. This is an honest and important result: **a method that shines on
synthetic data can be near-useless when the real-world signal is weak.**

### Lenta SMS RCT (moderate signal) — modeling + evaluation + **deployment**

| Learner | Qini score |
|---|---|
| **T-learner** | **0.156** |
| DR-learner | 0.049 |
| S-learner | −0.059 |
| X-learner | −0.092 |

<img src="images/lenta_qini.png" width="55%">

*The T-learner (red) rises well above random through the first half of the
population; the X-learner (yellow) stays below it.* Only the two simplest
constructions are clearly useful, and the elaborate **X-learner does worse than
random** — with ample samples in both arms, its cross-imputation machinery adds
noise instead of value.

**Deployment + validation (the extra step, unique to Lenta).** The best learner
was refit on the development set and used to score a fully **held-out 20%** the
model never trained on. Selecting the top 20% by predicted uplift and checking
their *real* response (using the held-out `T` and `Y`):

| Group | Actual uplift |
|---|---|
| **Selected top 20%** | **0.023** |
| The rest 80% | 0.007 |
| Whole population | 0.011 |

The targeted group's real uplift is **~2× the population average** and ~3× the
rest — using one-fifth of the reach. This closes the loop from *evaluation* to
an *honest business decision*: whom to actually contact.

## Conclusion

Across a strong-signal synthetic benchmark, a weak-signal email RCT, and a
moderate-signal SMS RCT, one pattern is consistent:

- **Simple learners (S, T) are the most robust.** They won or tied on every
  dataset.
- **Complex learners (X, DR) did not pay off** — X-learner even went *negative*
  on Lenta, and complex learners were more sensitive to sample size (DR went from
  −0.004 at 42k to +0.049 at 687k). Their theoretical advantages need conditions
  this data didn't stress.
- **Signal strength decides everything.** The same pipeline is genuinely useful
  on Lenta (targeting doubles uplift) and near-useless on Hillstrom (all ≈
  random). Knowing *when a method won't help* is as important as knowing when it
  will.

The practical takeaway: **match estimator complexity to the problem, and validate
it empirically rather than assuming more complex is better.**

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
  on a fully held-out 20%, the top-20% selected by the model showed about double
  the real uplift of the population average (0.023 vs. 0.011), validated with the
  held-out set's real `T` and `Y`.

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
├── uplift_lenta.py         # Lenta SMS RCT (moderate signal, targeting validated)
└── images/
    ├── hillstrom_qini.png  # Qini curves — Hillstrom (weak signal)
    └── lenta_qini.png      # Qini curves — Lenta (moderate signal)
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
