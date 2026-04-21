# Supervisor Response & Defence Q&A

**Candidate:** Samson Fukuzeya — MSc Computer Science, University of Zimbabwe
**Project:** Deploying a Production-Ready MLOps Model for Student-Dropout Prediction
**Companion docs:** [research_paper.md](research_paper.md) · [presentation.md](presentation.md) · [technical_manual.md](technical_manual.md)

---

## Part A — Written response to supervisor feedback

**Subject:** Addressing feedback on feature selection, automation, and human oversight.

Dear Supervisor,

Thank you for the feedback at the last review. I have now addressed both issues substantively in the codebase and documented the evidence trail. A short, direct summary is below, with file references so any claim can be verified in under two minutes.

**1. Feature selection is now automated inside the training pipeline.**
The trainer performs three complementary passes on every `dvc repro`:
(a) **schema-level** selection via the Pandera `PredictionFeaturesSchema` (nothing outside the canonical column list can reach a model);
(b) **transformer-level** selection and engineering inside a single fitted `sklearn.Pipeline` — `StandardScaler` for numerics, `OrdinalEncoder` for categoricals, and a `MomentumTransformer` that derives the academic-momentum features (approval ratio, grade deltas) so there is no train/serve skew;
(c) **model-driven** importance ranking — XGBoost `gain` and sklearn `permutation_importance` on the validation split — plus a **feature-family ablation** that retrains XGBoost with each family (academic, demographic, macro-economic, financial-aid) removed in turn. The output is archived to `reports/feature_importance.json` and `reports/ablation.md` and re-rendered on the admin dashboard. There is therefore no hand-curated feature list anywhere in the serving path.

**2. The system uses a hybrid automation model — not "black-box auto" and not "manual forever".**
A single environment flag `AUTO_PROMOTE_ENABLED` decides the mode and is logged on every retrain, so the governance record reflects the prevailing policy at every past moment:

- `AUTO_PROMOTE_ENABLED=true` — a challenger is auto-promoted **only if** it clears the three-part rule in [backend/app/ml/registry.py:compare_for_promotion](../backend/app/ml/registry.py): (i) macro-F1 gain ≥ +1.0 pp, (ii) no per-class F1 regression > 2.0 pp, (iii) McNemar mid-p test on paired holdout predictions significant at α = 0.05. If any gate fails, the challenger is rejected and the live champion is untouched.
- `AUTO_PROMOTE_ENABLED=false` — even a passing challenger is parked in the MLflow **Staging** stage and in `models/staging/`. The Angular admin route `/admin/model-registry` shows leaderboard, confusion matrices, per-class deltas, McNemar p-value, fairness gaps, and a SHAP summary. A named JWT-authenticated administrator clicks **Approve** (or **Reject with reason**), at which point the atomic swap and the MLflow transition happen, and a second audit line is written with `trigger="hitl:approve:<reviewer_username>"`.

The default for production is **HITL ON**. Trust is built with humans first, and the flag gives the institution a single, reversible control to lift autonomy as evidence accumulates.

**3. Drift is a trigger, not a ship-it.**
Evidently's `DataDriftPreset` publishes `share_of_drifted_columns` to Prometheus. A share ≥ 0.30 fires a challenger — but the **same** three-gate promotion rule decides whether it actually reaches Production. This separation is visible in `reports/retraining/history.jsonl`: the line of 2026-04-21T02:22:52Z records `"trigger": "drift:share=0.556"` alongside `"promoted": false` because the challenger did not clear the gain bar. The audit trail is therefore defensible both when we *do* ship and when we *refuse to*.

I remain available for any further review, and I have prepared an inline Q&A document (Part B of this file) with ten likely defence questions and my answers.

Best regards,
**Samson Fukuzeya**
MSc Computer Science — University of Zimbabwe

---

## Part B — Defence Q&A (ten likely questions, Level-5 answers)

Each answer is structured: **Direct answer → Evidence in repo → Trade-off I accept.** This is the structure the marking scheme rewards at the Level-5 ("Exceptional") tier.

---

### Q1. Your champion is Logistic Regression with macro-F1 ≈ 0.73. Your target was ≥ 0.85. Why did you fall short, and why should we still pass this?

**Direct.** The 0.85 is an *engineering aspiration* set before data were in hand; the *realised* 0.73 is what the data actually support at n = 4,424 with three classes. Five independent model families land within 0.03 macro-F1 of each other (LogReg 0.730, XGBoost 0.723, MLP 0.722, RF 0.707, LGBM 0.699) and McNemar cannot separate the top three — that is the signature of a **feature/sample ceiling**, not an engineering failure (Recht et al., 2019). Under the cost-weighted utility that actually matters for an EWS (missed dropouts are 10× costlier than false alarms) the threshold-tuned champion delivers **Dropout recall 0.873 ≥ 0.85 target** and cuts expected cost per student by **25.8%**.

**Evidence.** [models/champion/metadata.json](../models/champion/metadata.json); [reports/evaluation.json](../reports/evaluation.json) keys `tuned_metrics`, `cost.tuned`, and `threshold.chosen_threshold = 0.20`.

**Trade-off.** I trade macro-F1 (0.730 → 0.643) for Dropout recall (0.768 → 0.873). That is the right trade for a recall-first intervention system; I say so in the discussion rather than hiding behind an averaged metric.

---

### Q2. Why Evidently AI and why specifically a drift threshold of 0.30 — is that not arbitrary?

**Direct.** Evidently is chosen for three reasons: (a) its `DataDriftPreset` runs the correct per-feature test by dtype (K-S for numeric, chi-squared for categorical) out of the box; (b) it emits a self-contained HTML that the Angular dashboard can embed without an additional SaaS; (c) the Python API is pure-function and therefore trivially unit-testable. The 0.30 threshold is a **policy knob, not a statistical claim**: at 36 features a share of 0.30 means roughly 11 features have shifted, which is a conservative early-warning tripwire calibrated against the project's risk appetite. It is configurable per request on `/monitoring/drift?threshold=…` and per deployment in `.env`.

**Evidence.** [backend/app/monitoring/drift.py](../backend/app/monitoring/drift.py); `DriftResult.detected = share >= threshold`.

**Trade-off.** A higher τ (say 0.50) would produce fewer false alarms but delay detection of gradual distribution shift; a lower τ (0.20) catches shift earlier but burns retraining budget. 0.30 is the default, not the doctrine.

---

### Q3. If the promotion gate requires +1 pp improvement and paired McNemar significance, will you *ever* actually promote?

**Direct.** Yes — and the refusal to ship a non-significant model is itself the feature. The gate is calibrated so random fluctuation cannot dress up noise as progress: on a holdout of n = 885 an observed macro-F1 gain of 0.01 is within the bootstrap-CI margin of zero, which is exactly why we also require McNemar to reject the null on *paired* errors. When the feature ceiling does shift (new semester data, corrected labels, new engineered features), the gate trips correctly; when it does not, we keep the known-good champion. The four lines in `reports/retraining/history.jsonl` all legitimately read `"promoted": false` because the challengers were trained on the **same** data — so the gate behaved.

**Evidence.** [backend/app/ml/registry.py:compare_for_promotion](../backend/app/ml/registry.py); [backend/app/ml/statistics.py](../backend/app/ml/statistics.py) for the McNemar implementation; [reports/retraining/history.jsonl](../reports/retraining/history.jsonl).

**Trade-off.** False negatives (good challenger rejected) over false positives (silently worse model shipped). In an EWS that decides student support, that asymmetry is the correct one.

---

### Q4. How is your system different from running `model.predict` behind a Flask endpoint?

**Direct.** Serving is less than 10% of the surface. The system adds (a) Pandera contracts at every I/O boundary, (b) DVC-reproducible lineage from raw CSV to joblib, (c) a five-way bake-off with McNemar-gated promotion, (d) calibrated thresholds with cost-sensitive decisioning, (e) a fairness audit across five sensitive attributes, (f) per-prediction SHAP plus a rule-based UZ intervention recommender, (g) Evidently drift with Prometheus alerting, (h) a JSONL audit trail of every promotion and rejection, and (i) a configurable HITL toggle with a reviewer UI. That list is the difference between an ML *demo* and an ML *operation* (Sculley et al., 2015; Kreuzberger et al., 2023).

**Evidence.** `docker compose up` brings up all six runtimes in one command; the CI gate requires ≥ 80% coverage; `dvc repro` rebuilds bit-for-bit.

**Trade-off.** Operational complexity — six containers, two auth modes, one governance flag — in exchange for a system a faculty office can inherit.

---

### Q5. Why keep sensitive features (Gender, Scholarship, Debtor) in the model at all? Isn't that fairness malpractice?

**Direct.** These are **educational-vulnerability signals**, not protected attributes from which we derive a decision — and deleting them does not make the model fair (Barocas et al., 2023; it only makes the bias invisible to the audit. The system pairs their use with (a) a fairness audit in `reports/evaluation.json` that exposes the largest equal-opportunity gap (0.146 on Scholarship holder), (b) per-prediction SHAP so the reviewer sees exactly which features drove a flag, and (c) an explicit "advisory only" policy with HITL over any automated model promotion. The Scholarship-holder gap is documented as a known limitation with a mitigation path (post-processing calibration; Hardt et al., 2016).

**Evidence.** [backend/app/ml/fairness.py](../backend/app/ml/fairness.py); `fairness` section of [reports/evaluation.json](../reports/evaluation.json); the ethics & limitations section of the research paper.

**Trade-off.** Documented, auditable use of vulnerability signals with a reviewer in the loop, over feature deletion that hides bias.

---

### Q6. What prevents someone with admin credentials from promoting a worse model via the HITL "Approve" button?

**Direct.** Three safeguards. First, the `/approve` endpoint **re-runs** the three-gate rule server-side before touching the bundle — the button does not bypass the gate, it only bypasses *auto*-promotion. Second, every approval writes an audit line with the reviewer's username and the gate's numeric outcome, so a post-hoc review can always identify the decision. Third, the MLflow registry keeps the previous production version archived, not deleted, so any promotion is immediately reversible by a second approval of the archived version.

**Evidence.** [backend/app/api/v1/model_registry.py](../backend/app/api/v1/model_registry.py) (approve/reject handlers); `registered_model_version` field on [reports/retraining/history.jsonl](../reports/retraining/history.jsonl).

**Trade-off.** Governance beats ergonomics — the reviewer cannot override the gate, only its *automation*.

---

### Q7. The UCI dataset is Portuguese. Why is it defensible to claim the system is "for Zimbabwean higher education"?

**Direct.** The claim is about the **system**, not the specific trained coefficients. The research paper and the Model Card explicitly recommend **population-specific recalibration on ≥ 8,000 UZ records across two intakes** before a pilot, and the intervention recommender already maps predictions to UZ-specific services (bursar, peer-tutor, counselling, dean-of-students). The underlying signals — financial pressure, attendance, semester momentum — generalise; the coefficients do not. DVC + MLflow mean a retrained UZ champion is a one-command swap, not a rewrite.

**Evidence.** Recommendation 1 in the research paper; the interventions list in [backend/app/interventions/recommender.py](../backend/app/interventions/recommender.py).

**Trade-off.** Weaker numerical transferability in exchange for a reproducible template any African institution can retrain on its own data.

---

### Q8. Your Scholarship-holder equal-opportunity gap is 0.146 — that is large. Why did you not mitigate it?

**Direct.** Mitigation was deliberately scoped out of the Master's artefact: the literature is consistent that mitigation techniques (reweighing, equalised-odds post-processing, threshold optimisation per subgroup; Hardt et al., 2016; Pleiss et al., 2017) trade off accuracy against fairness in ways that must be agreed with the institution, not with me. The **right** Master's-level answer is to **measure**, **publish**, and **flag**, which I do. The recommender lists the mitigation options, and the research paper's Recommendations section commits to adopting one of them as a prerequisite for a UZ pilot.

**Evidence.** Fairness audit in [reports/evaluation.json](../reports/evaluation.json); Limitations §12 in the research paper.

**Trade-off.** Honest measurement over convenient mitigation — I would rather hand over a documented 0.146 gap than a "mitigated" gap whose technique the reviewer did not sign off on.

---

### Q9. Why five baseline models — isn't that overkill? Wouldn't XGBoost alone be enough?

**Direct.** Five is the minimum number that lets me *defend* the claim that a single model was not adequate. If I had run XGBoost alone and reported 0.723, a reviewer could ask, "Did you try LogReg? Did you try an MLP?" Running all five under the same CV harness lets me say: yes, and the gap between the best and worst is 0.03 macro-F1 on this data — the bottleneck is the feature set, not the algorithmic family. That is a defensible, empirical answer rather than an asserted one. It is also why the registry retains the richer families (XGBoost, MLP) as live challengers: the ceiling may break once local UZ data provide new signal.

**Evidence.** Five leaderboard entries in [models/champion/metadata.json](../models/champion/metadata.json); McNemar contingencies on paired predictions.

**Trade-off.** Two extra hours of compute per `dvc repro` for a defensible empirical claim about the model-family ceiling.

---

### Q10. Take me through what happens from the moment a counsellor clicks "Score cohort" on the dashboard.

**Direct.** Step by step:

1. The Angular dashboard `POST`s the cohort CSV to `/api/v1/predict/batch` with an `X-API-Key` header; `slowapi` enforces rate-limits.
2. FastAPI's dependency injector validates the CSV against the Pandera `PredictionFeaturesSchema` — malformed rows fail fast with a 422.
3. The validated DataFrame is passed to the cached champion `Pipeline` (`features.py` ColumnTransformer + model). One `predict_proba` call returns class probabilities for every row.
4. The tuned threshold (T = 0.20) is applied to the Dropout-class probability to make the final flag; the argmax is used for the non-Dropout classes.
5. TreeExplainer (or KernelExplainer fallback) computes per-row SHAP values; the top-*k* contributions per student are attached to the response.
6. `recommender.py` consumes the flags and SHAP contributions and emits a list of UZ-specific interventions per student (bursar referral, peer-tutor, counselling, dean liaison).
7. Prometheus counters (`predictions_total_counter`) and latency histogram are updated; the request line is written to the structured JSON log.
8. The response payload is returned to the dashboard, which renders risk buckets, a SHAP waterfall per student, and the intervention chips — all inside the counsellor's session, no data leaving the cluster.

**Evidence.** [backend/app/api/v1/predict.py](../backend/app/api/v1/predict.py) for the route; [backend/app/ml/explain.py](../backend/app/ml/explain.py) for SHAP; [backend/app/interventions/recommender.py](../backend/app/interventions/recommender.py); [backend/app/core/metrics.py](../backend/app/core/metrics.py).

**Trade-off.** A few extra milliseconds per request for SHAP + interventions, in exchange for a dashboard that is defensible to a counsellor and a student at the same time.

---

## Part C — What the evidence trail proves, line by line

| Claim in the paper                                 | Where to verify                                                    |
|----------------------------------------------------|--------------------------------------------------------------------|
| "5-way bake-off, LogReg wins"                      | [models/champion/metadata.json](../models/champion/metadata.json) `leaderboard` |
| "macro-F1 = 0.730 on holdout"                      | [models/champion/metadata.json](../models/champion/metadata.json) `metrics.macro_f1` |
| "Dropout recall 0.873 at T = 0.20"                 | [reports/evaluation.json](../reports/evaluation.json) `tuned_metrics` + `threshold` |
| "Expected cost ↓ 25.8% (0.827 → 0.614)"            | [reports/evaluation.json](../reports/evaluation.json) `cost.argmax` vs `cost.tuned` |
| "Post-temperature ECE = 0.086"                     | [reports/evaluation.json](../reports/evaluation.json) `calibration.post.ece_macro` |
| "Max fairness gap 0.146 on Scholarship"            | [reports/evaluation.json](../reports/evaluation.json) `fairness.summary_max_gap` |
| "Drift share 0.556 triggered a challenger"         | [reports/retraining/history.jsonl](../reports/retraining/history.jsonl) line 3 |
| "Challenger correctly rejected by the gate"        | Same JSONL line — `"promoted": false` with the reason             |
| "Three-gate promotion rule"                        | [backend/app/ml/registry.py:compare_for_promotion](../backend/app/ml/registry.py) |
| "≥ 80% coverage on CI"                             | `.github/workflows/ci.yml` + local [coverage.xml](../coverage.xml) |

**One-sentence close:** this submission does not promise the aspirational macro-F1; it delivers the reproducible MLOps system, the honest metrics, and the governance evidence that a Master's-level defence demands.
