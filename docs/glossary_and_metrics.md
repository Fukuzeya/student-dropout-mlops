# Plain-English Glossary & Why We Picked These Metrics
## A walk-through for teammates joining the project

**Audience.** Project teammates who are comfortable with Python but are *not* ML specialists. The goal of this document is simple: by the time you finish reading it, you should be able to sit in a code review and understand every acronym on the dashboard and every column in `reports/evaluation.json` — and you should be able to explain in your own words why we *do not* report accuracy.

Companion docs: [research_paper.md](research_paper.md) · [technical_manual.md](technical_manual.md) · [supervisor_qa.md](supervisor_qa.md).

---

## Part 1 — The glossary

### 1.1 Core project jargon

| Term | Plain-English meaning | Where we use it |
|------|-----------------------|-----------------|
| **MLOps** | "DevOps for ML". The set of practices that turn a notebook model into a service you can deploy, monitor, retrain, and audit. | The whole repo. |
| **Champion** | The model currently serving production predictions. | `models/champion/model.joblib` |
| **Challenger** | A newly trained model competing to replace the champion. Only promoted if it beats the champion under a strict rule. | `models/staging/model.joblib` |
| **Promotion gate** | The set of rules that decide whether a challenger replaces the champion. Ours has three rules. | `backend/app/ml/registry.py:compare_for_promotion` |
| **Stratified split** | When we split data into train/val/test, we keep the class proportions the same in every split. Prevents accidentally putting all Dropouts in one bucket. | `backend/app/ml/train.py` (preprocess stage) |
| **Cross-validation (CV)** | Training & evaluating the model on several train/val splits in turn, so the reported score is not a lucky one-off. We use **5-fold stratified CV**. | `params.yaml` → `cv.n_splits: 5` |
| **Holdout** | A final test set the model *never* sees during training or tuning. We use a **20%** holdout, never touched until the final report. | `data/processed/test.parquet` |
| **Bake-off** | Training several algorithms under identical conditions and letting the numbers pick the winner. We bake-off five models. | `reports/evaluation.json` + `models/champion/metadata.json` |
| **Pandera** | A Python library that validates DataFrames against a schema. If a row is malformed, it never reaches the model. | `backend/app/ml/schemas.py` |
| **DVC** | Data Version Control. Git for data and models. Ensures `dvc repro` on a clean clone produces the same numbers. | `dvc.yaml`, `params.yaml` |
| **MLflow** | Experiment tracker + model registry. Every training run is logged with params, metrics, and artefacts. | `mlruns/`, port 5000 |
| **SHAP** | SHapley Additive exPlanations. Per-prediction feature attributions — "why did the model flag *this* student?" | `backend/app/ml/explain.py` |
| **Evidently AI** | Library that produces data-drift reports by comparing a production batch to the training distribution. | `backend/app/monitoring/drift.py` |
| **Prometheus / Grafana** | Metrics collector and dashboard. Shows request latency, volume, live macro-F1, drift share. | `infrastructure/` |
| **HITL (Human-in-the-Loop)** | A policy where a human must approve an action before it takes effect. In our case: approving a model promotion. | `AUTO_PROMOTE_ENABLED` flag |

### 1.2 The dataset and classes

| Term | Plain-English meaning |
|------|-----------------------|
| **Dropout** | Target class 1. The student left the programme without graduating. **This is the class we care most about catching.** |
| **Enrolled** | Target class 2. The student is still enrolled at the snapshot date — outcome uncertain. This is the hardest class to predict because it is a "transition" class. |
| **Graduate** | Target class 3. The student successfully completed. |
| **Class imbalance** | The three classes are not equal in size: Graduate ≈ 50%, Dropout ≈ 32%, Enrolled ≈ 18%. Some metrics (like accuracy) look artificially good on imbalanced data. |
| **Academic momentum** | Our engineered features: credits *approved* divided by credits *enrolled* per semester, grade deltas across semesters. Captures whether a student is gaining or losing ground. |
| **Cold start** | The problem of predicting for a first-week student who does not yet have any curricular features. Our system acknowledges this in its limitations. |

### 1.3 The MLOps lifecycle vocabulary

| Term | Plain-English meaning |
|------|-----------------------|
| **Reproducibility** | Someone else can check out the repo, run one command, and get the exact same numbers we got. |
| **Lineage** | The ability to trace a prediction back to the exact data, code, and hyperparameters that produced it. We have it end-to-end. |
| **Drift** | Real-world data starts to look different from the data the model was trained on (a new syllabus, a fees policy change). We watch for it with Evidently. |
| **Drift share** | The proportion of features that have statistically shifted between reference and current data. A number between 0.0 and 1.0. We alert at **≥ 0.30**. |
| **Atomic swap** | Replacing the live model file in a single filesystem operation so the API never sees a half-written model. We use `Path.replace()` (POSIX rename). |
| **Audit trail** | A permanent, append-only log of every promotion decision. Lives in `reports/retraining/history.jsonl`. |
| **Rollback** | Reverting to a previous model if the new one turns out to be worse in production. Free for us: MLflow archives the old version. |

---

## Part 2 — The statistics glossary (in plain English)

These are the numbers you will see in `reports/evaluation.json` and on the dashboard. Every definition is paired with a *one-line example* from our real results.

### 2.1 The confusion matrix — the source of truth

A 3×3 grid for our three classes. Rows = true class, columns = predicted class. Example from our champion:

|             | → pred Dropout | → pred Enrolled | → pred Graduate |
|-------------|---------------:|----------------:|----------------:|
| **true Dropout**  | 109 | 21 | 12 |
| **true Enrolled** |   9 | 51 | 20 |
| **true Graduate** |  14 | 28 | 179 |

Every metric below is computed from this grid. Read one row and you understand the model's behaviour on one class.

### 2.2 Precision, recall, and F1 — for ONE class

For a single class (let's say Dropout):

- **TP** (true positive) = predicted Dropout and actually Dropout. In our matrix: **109**.
- **FP** (false positive) = predicted Dropout but actually not. In our matrix: 9 + 14 = **23**.
- **FN** (false negative) = predicted not Dropout but actually was. In our matrix: 21 + 12 = **33**.

From these three:

- **Precision** = TP / (TP + FP) = 109 / (109 + 23) = **0.826**.
  *"Of the students we flagged as Dropout, 82.6% actually were."*
- **Recall** = TP / (TP + FN) = 109 / (109 + 33) = **0.768**.
  *"Of all the actual Dropouts, we caught 76.8%."*
- **F1** = 2 · (precision · recall) / (precision + recall) = **0.796**.
  *"A harmonic mean of the two. Low if either is low."*

The harmonic mean matters: you cannot get a high F1 by sacrificing precision for recall or vice-versa. Both have to be reasonably good.

### 2.3 Macro-F1 vs weighted-F1 vs accuracy

This is the section you will quote in the defence. Read it twice.

- **Accuracy** = (total correct predictions) / (total predictions). Treats every row equally.
- **Weighted-F1** = average of per-class F1, weighted by each class's *support* (how many examples it has). Large classes dominate.
- **Macro-F1** = plain average of per-class F1. **Every class counts equally**, regardless of its size.

On our data: Graduate is ~50% of the test set. If the model just predicted "Graduate" for everyone, it would score:

- accuracy ≈ **50%** (sounds decent),
- weighted-F1 ≈ 0.33 (drops, because it sees per-class F1),
- **macro-F1 ≈ 0.22** (collapses — because the F1 for the two ignored classes is zero).

**That is why we report macro-F1 as the primary metric.** It refuses to let us pat ourselves on the back for ignoring the Dropout class.

### 2.4 Macro-AUC OVR

**AUC** = Area Under the ROC Curve — the probability that a random positive is ranked above a random negative. **OVR** = One-vs-Rest, because we have three classes so we average three per-class AUCs.

Example from our champion: macro-AUC OVR = **0.891**. This tells us the model's *ranking* of students by risk is strong, even if its *thresholded decisions* (macro-F1 = 0.730) are weaker. The gap between 0.891 and 0.730 is exactly what threshold tuning exploits.

### 2.5 Bootstrap confidence intervals

When a single holdout number looks like **macro-F1 = 0.689**, we do not know how noisy that number is. Bootstrap = resample the holdout 1,000 times with replacement, recompute the metric each time, take the 2.5th and 97.5th percentiles. That gives us a **95% confidence interval**.

Our champion: macro-F1 = 0.689, **95% CI = [0.655, 0.719]**. We can say "the true macro-F1 is plausibly between 0.655 and 0.719" and *not* mistake 0.002 differences between models for real differences.

### 2.6 McNemar's test

When comparing **two** models on the **same** test set, the right test is not a two-sample t-test — it is McNemar's test on paired disagreements:

- **b** = rows where Model A was right and Model B was wrong.
- **c** = rows where Model B was right and Model A was wrong.

If b and c are very different, one model is genuinely better. If they are similar, the difference is noise. We use **McNemar's mid-p** variant (safer for small b+c) with α = 0.05. This is the **third gate** in our promotion rule.

### 2.7 Calibration (ECE & temperature scaling)

A model outputs a probability. If it says "0.8 probability of Dropout" for a hundred students, we want **80 of them** to actually be Dropout. If only 60 are, the model is **overconfident**.

- **Expected Calibration Error (ECE)** = bin the probabilities (we use 15 bins), measure the gap between predicted probability and observed frequency in each bin, average the gaps. Lower is better.
- **Temperature scaling** = one learned scalar T that we divide the logits by, softening or sharpening the probabilities. Cheap, post-hoc, and usually makes ECE better.

Our champion: pre-scaling ECE = 0.083; post-scaling ECE = **0.086** with T = 1.19. Here the improvement is marginal — but we report both numbers because hiding the "before" would be dishonest.

### 2.8 Cost matrix and expected cost

Not all mistakes are equal. Our cost matrix encodes:

|                  | pred Dropout | pred Enrolled | pred Graduate |
|------------------|-------------:|--------------:|--------------:|
| **true Dropout**  |    0.0  |     5.0   |    **10.0**    |
| **true Enrolled** |    1.5  |     0.0   |     2.5       |
| **true Graduate** |    1.0  |     0.5   |     0.0       |

Reading row 1: predicting "Graduate" when the truth is "Dropout" costs **10 units** — because we missed a student who needed help. The mirror error (predicting "Dropout" when the truth is "Graduate") costs **1 unit** — we may have flagged a thriving student, which is mildly annoying but cheaply reversible.

- **Expected cost per student** = sum of (confusion-matrix cell × cost) / total.

Ours drops from **0.827 at argmax** to **0.614 at tuned threshold** — a **25.8% cost reduction**. This is the metric that actually matches policy.

### 2.9 Fairness gaps

Computed per sensitive attribute (Gender, Age band, Scholarship holder, Debtor, International):

- **Demographic-parity gap** = difference between groups in the rate at which they are flagged Dropout.
- **Equal-opportunity gap** = difference between groups in Dropout **recall**. "Does the model catch an at-risk scholarship student as well as an at-risk non-scholarship student?"
- **Predictive-equality gap** = difference between groups in Dropout **false-positive rate**.

Our largest: Scholarship-holder equal-opportunity gap = **0.146** (non-scholarship 0.883 vs scholarship 0.737). Flagged as a limitation.

---

## Part 3 — Why we do NOT use accuracy

This is the question every non-ML teammate asks. The short answer: **accuracy is misleading on imbalanced, cost-sensitive, multi-class data — which is exactly what we have.**

### 3.1 The imbalance argument

Our test set has ~50% Graduates. A model that predicts "Graduate" for everyone scores **accuracy ≈ 50%** — but its Dropout recall is **0.0**, meaning it catches zero at-risk students. That is a completely useless EWS that nonetheless looks half-decent on accuracy. Accuracy does not *require* the model to be good at any particular class; it only requires it to be good at the *most common* one.

### 3.2 The cost argument

Accuracy treats all mistakes equally. In an EWS, they are not equal — missing a Dropout is ten times costlier than a false alarm. A model that is 92% accurate by making its mistakes on the *expensive* class (Dropout) is worse than an 88%-accurate model that makes its mistakes on the *cheap* class (Graduate). Accuracy cannot see the difference; expected cost per student can.

### 3.3 The operational argument

The intervention budget at any university is finite. The people using this system — academic-affairs officers, counsellors — care about one question: *"Of the students I should be helping, how many did you surface?"* That is Dropout **recall**, not accuracy.

### 3.4 The academic-standards argument

Every modern reference in this field — Albreiki et al. (2021), Nagy & Molontay (2024), Bañeres et al. (2023), Moreno-Marcos et al. (2019) — reports F1, recall, and AUC, and explicitly warns against accuracy on imbalanced classification tasks. Using accuracy as a primary metric would invite exactly the kind of viva question we want to avoid.

---

## Part 4 — Why macro-F1 AND Dropout recall (not one or the other)

We use **two** primary metrics, with a tie-break rule. Here is the reasoning.

### 4.1 Why macro-F1 is the headline

- It forces the model to be good at **every** class, including the small Enrolled class.
- It combines precision and recall into a single interpretable number per class.
- It does not reward ignoring a minority class.
- It is the standard in the recent dropout-prediction literature.

### 4.2 Why Dropout recall is the tie-break

Macro-F1 alone is still a general-purpose metric; it does not know that *Dropout* is the class whose miss-cost is highest. Two models can have identical macro-F1 but very different Dropout recalls — one misses 3% of at-risk students and the other misses 20%. For our EWS, that difference is everything.

The **selection rule** is therefore:

1. Pick the model with the highest holdout macro-F1.
2. If two models are tied within the noise floor (McNemar cannot separate them), pick the one with the higher **Dropout recall**.

This is why `params.yaml` declares both:

```yaml
evaluation:
  primary_metric: macro_f1
  secondary_metric: dropout_recall
```

### 4.3 Why we *also* report weighted-F1, AUC, ECE, cost and fairness

Because no single number describes model quality. The dashboard and the evaluation report deliberately over-provide: macro-F1 for class-balanced quality, Dropout recall for operational utility, weighted-F1 for a sanity check against the literature, AUC for the ranking quality, ECE for the probability quality, cost per student for the policy match, and fairness gaps for the ethical contract. A reviewer who only trusts one of those numbers can pick their favourite; a reviewer who reads all of them gets the whole picture.

---

## Part 5 — A one-page cheat sheet for the defence

Write this on a sticky note if you have to:

- **Primary metric:** macro-F1 — refuses to let us hide bad Dropout performance behind class imbalance.
- **Operational metric:** Dropout recall ≥ 0.85 — forced by the threshold sweep.
- **Policy metric:** expected cost per student — 25.8% reduction from argmax to tuned.
- **Ranking metric:** macro-AUC OVR (0.891) — tells us the model's *ranking* is stronger than its *thresholded decisions*.
- **Calibration:** post-temperature ECE = 0.086 — the probabilities are trustable to about that margin.
- **Significance:** McNemar's mid-p on paired predictions — no hand-waving "this model looks better".
- **Why not accuracy?** Imbalance + cost-asymmetry + operational reality + every modern reference in this literature.

If someone asks "can you summarise all your metrics in one sentence?", answer:

> We report **macro-F1** because the classes are imbalanced and every class matters, **Dropout recall** because missed dropouts are ten times costlier than false alarms, **expected cost** because policy decisions happen in units of effort not percentages, and **fairness gaps** because a recall improvement we earned by being worse on scholarship students would not be an improvement at all.

Anything else is decoration.
