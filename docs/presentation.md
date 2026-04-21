---
marp: true
theme: default
paginate: true
title: Production-Ready MLOps for Student-Dropout Prediction
author: Samson Fukuzeya — MSc CS, University of Zimbabwe
---

<!-- _class: lead -->

# Deploying a Production-Ready Model Using MLOps
## Early Student-Dropout Prediction for Zimbabwean Higher Education

**Samson Fukuzeya**
MSc Computer Science — University of Zimbabwe
Supervisor defence · 21 April 2026

---

# 1 · The Zimbabwean problem we are solving

- **20–35%** four-to-six-year attrition in Zimbabwean public universities (MHTEISTD, 2023).
- **No** operating data-driven early-warning system at any Zimbabwean institution today.
- Current practice: *end-of-semester* transcript review — after intervention is still cheap.
- The cost is paid four times: by the **student, institution, funder, labour market**.
- Our thesis: the bottleneck is **not an algorithm** — it is a reproducible, governable, ethical MLOps pipeline a single office can operate.

---

# 2 · Research questions

- **RQ1.** Does a disciplined **5-model bake-off** + formal promotion rule beat single-algorithm baselines on UCI dropout data?
- **RQ2.** What **calibrated threshold** maximises Dropout recall without crashing macro-F1, and does temperature scaling help?
- **RQ3.** Can a single researcher, on free and open-source tooling only, stand up an EWS that passes reproducibility, governance and fairness checks?

**Target metrics:** macro-F1 **≥ 0.85** (aspirational); Dropout recall **≥ 0.85** (mandatory).

---

# 3 · System architecture (the "How")

```
 Angular + Tailwind  ──HTTPS+JWT──▶  FastAPI
 risk dashboard                      /predict  /predict/batch
 SHAP waterfalls                     /monitoring/drift  /retrain
        │                             │        │        │
        │                        Pandera  Prometheus   MLflow
        │                        contracts /metrics  tracking+registry
        ▼                             │        │        │
 Docker Compose   ──────▶ champion pipeline (joblib) ◀─── DVC lineage
 (8000,5000,9090,3000)          SHAP · interventions · Evidently drift
```

**Stack:** FastAPI · Angular · Tailwind · Pandera · MLflow · DVC · Prometheus · Grafana · Evidently AI · Docker Compose · GitHub Actions.

---

# 4 · Data strategy & ethics

- **UCI "Predict Students' Dropout and Academic Success"** — 4,424 rows × 36 features, 3 classes (Dropout / Enrolled / Graduate).
- **Pandera** `DataFrameModel` validates **every** I/O boundary (training, batch, `/predict`).
- **SHAP** per-prediction attributions returned with every risk score.
- **Fairness audit** across Gender, Age band, Scholarship, Debtor, International.
- Predictions are **advisory only** — the system never auto-actions a student.

---

# 5 · Champion–Challenger methodology

- Five families under the **same** stratified 5-fold CV + frozen holdout:
  **LogReg · RandomForest · XGBoost · LightGBM · PyTorch MLP**.
- **Selection rule:** highest macro-F1, tie-break on Dropout recall.
- **Promotion gate** (`backend/app/ml/registry.py:compare_for_promotion`):
  1. macro-F1 gain **≥ +1.0 pp**, **and**
  2. no per-class F1 regression **> 2.0 pp**, **and**
  3. McNemar mid-p test **p < 0.05** on paired holdout predictions.
- Every run logged to **MLflow**; every promotion appended to `reports/retraining/history.jsonl`.

---

# 6 · Results — the honest leaderboard

| Rank | Model | Macro-F1 | Dropout recall |
|---:|---|---:|---:|
| 1 | **Logistic Regression** (champion) | **0.730** | 0.768 |
| 2 | XGBoost | 0.723 | 0.775 |
| 3 | PyTorch MLP | 0.722 | 0.754 |
| 4 | Random Forest | 0.707 | 0.746 |
| 5 | LightGBM | 0.699 | 0.782 |

- Threshold-tuned Dropout recall **= 0.873 ≥ 0.85 target** at T = 0.20.
- Post-temperature **ECE 0.086** · cost per student **↓ 25.8%**.
- Aspirational macro-F1 ≥ 0.85 **not yet reached** — feature/sample ceiling on 4,424 rows.

---

# 7 · Automation vs Human-in-the-Loop
## (Direct answer to supervisor's feedback)

- **Feature selection is automated:** XGBoost gain / permutation importance inside the trainer + a feature-group ablation report.
- **Retraining is automated:** Evidently drift share ≥ 0.30 or scheduled cron fires a DVC-clean challenger run.
- **Promotion is the supervised step.** A single flag `AUTO_PROMOTE_ENABLED` decides:
  - `true`  → challenger auto-promoted **only** if the three-gate rule passes.
  - `false` → challenger parked in **Staging**; a JWT-authenticated admin reviews leaderboard + SHAP + fairness before approving via the Angular dashboard.
- **Default for production: HITL ON.** Trust is built with humans first, lifted later.

---

# 8 · Closed-loop monitoring

- `/monitoring/drift` → Evidently `DataDriftPreset` vs the frozen `data/reference/reference.parquet` snapshot.
- `share_of_drifted_columns` published as a Prometheus gauge; Grafana renders it alongside API latency, request volume, live macro-F1.
- A drift event does **not** auto-ship a new model — it triggers a challenger; only the promotion gate can ship.
- Two production-shaped batches tested; a synthetic perturbation with drift share **0.556** correctly fired a challenger run that was then **rejected** by the gate.

---

# 9 · Reproducibility & engineering evidence

- **One command** rebuilds everything: `dvc repro && docker compose up --build`.
- **CI gate**: ruff + mypy + pytest with **≥ 80% coverage** on every push.
- **Versioning**: git (code) + DVC (data & models) + MLflow (experiments & registry).
- **Auditability**: every promotion writes a JSONL audit entry with McNemar p-value, per-class deltas, trigger reason, n_test.
- **Containers**: FastAPI, MLflow, Prometheus, Grafana, Angular — all in one `docker-compose.yml`.

---

# 10 · Limitations (honest account)

- UCI data are **Portuguese** — coefficients do not transfer 1:1 to UZ cohort.
- n = 4,424 gives wide bootstrap 95% CI on macro-F1 ≈ **[0.655, 0.719]**.
- **Cold-start** for first-week students: curricular features are empty.
- **Fairness tax:** Scholarship-holder equal-opportunity gap **0.146** is flagged, not yet mitigated.
- Single-researcher delivery — no enterprise IAM / multi-tenant isolation.

---

# 11 · Recommendations & future work

- Local recalibration on ≥ 8,000 UZ records across two intakes.
- Integrate `/predict/batch` behind UZ SIS identity.
- Ship with **HITL ON** for the first three promotions, then reconsider.
- Publish a **Model Card** (Mitchell et al., 2019) and **Datasheet** (Gebru et al., 2021).
- Future: discrete-time survival head; federated learning across UZ · NUST · MSU · CUT; causal intervention trial; fairness-regression clause in the promotion gate.

---

<!-- _class: lead -->

# 12 · Thank you — ready for Q&A

**Live demo:** `http://localhost:8000/docs` · `http://localhost:5000` (MLflow) · `http://localhost:3000` (Grafana)

**Evidence trail:** `reports/evaluation.json` · `models/champion/metadata.json` · `reports/retraining/history.jsonl`

**Code:** `github.com/<repo>/student-dropout-mlops` — tagged `v1.0-defence`.

*"A missed dropout is the costly mistake — we optimise for recall, and we always keep a human in the loop."*
