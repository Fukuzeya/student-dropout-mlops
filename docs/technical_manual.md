# Technical System Manual
## Student-Dropout Prediction MLOps Platform

**Audience:** reviewers, operators, and engineers who need to understand *how* the system works end-to-end.
**Companion documents:** [research_paper.md](research_paper.md), [presentation.md](presentation.md), [supervisor_qa.md](supervisor_qa.md).
**Version:** v1.0-defence · 2026-04-21.

---

## 1. System overview

```
 ┌────────────────────────┐          ┌───────────────────────────┐
 │  Angular + Tailwind    │  HTTPS   │        FastAPI            │
 │  risk dashboard        │─────────▶│  /predict  /predict/batch │
 │  SHAP waterfalls       │   JWT    │  /monitoring/drift        │
 │  admin · HITL review   │          │  /retrain  /model/registry│
 └──────────┬─────────────┘          └───────┬──────────┬────────┘
            │                                │          │
            │                            Pandera    Prometheus
            │                            contracts  /metrics
            │                                │          │
            │                        ┌───────▼──────────▼────────┐
            │                        │  champion pipeline         │
            │                        │  (sklearn Pipeline, joblib)│
            │                        └──────┬────────────┬────────┘
            │                               │            │
            │                             SHAP       Evidently
            │                                          drift
            │                                            │
            │                                    ┌───────▼────────┐
            └───────────── MLflow ◀─────── DVC ──┤   retraining   │
                           registry       lineage│     loop       │
                                                 └────────────────┘
```

All six runtimes ship in one `docker-compose.yml`:

| Service    | Image source                  | Port  | Purpose                                         |
|------------|-------------------------------|-------|-------------------------------------------------|
| api        | `backend/Dockerfile`          | 8000  | FastAPI inference + retrain + drift             |
| frontend   | `frontend/Dockerfile`         | 4200  | Angular risk dashboard                          |
| mlflow     | `ghcr.io/mlflow/mlflow`       | 5000  | Experiment tracking + Model Registry UI         |
| prometheus | `prom/prometheus`             | 9090  | Scrapes `/metrics` every 15s                    |
| grafana    | `grafana/grafana`             | 3000  | Dashboards (latency · volume · macro-F1 · drift)|
| evidently  | rendered static HTML          | n/a   | Drift reports under `reports/drift/drift_*.html`|

---

## 2. The DVC training pipeline

`dvc repro` walks a deterministic DAG (`dvc.yaml`):

1. `download`   → pulls UCI CSV to `data/raw/dropout.csv`.
2. `validate`   → Pandera `RawStudentSchema` → `data/processed/validated.parquet`.
3. `preprocess` → 70/10/20 stratified split (seed 42) → train/val/test parquets **and** `data/reference/reference.parquet` (the Evidently anchor).
4. `train`      → runs the 5-way bake-off, picks the champion, persists `models/champion/{model.joblib, metadata.json}` **and** logs every run to MLflow.
5. `evaluate`   → bootstrap CIs, temperature calibration, threshold sweep, cost-sensitive decisioning, fairness audit → `reports/evaluation.json` + `reports/figures/*.png`.

Every stage declares its `deps`, `params`, and `outs` — so a single changed hyperparameter in `params.yaml` invalidates exactly the downstream stages it touches, and nothing else.

---

## 3. Automated feature selection

Feature selection is **pipeline-internal** and automatic. It happens in three passes inside [backend/app/ml/train.py](../backend/app/ml/train.py) and [backend/app/ml/features.py](../backend/app/ml/features.py):

### 3.1 Schema-level selection (Pandera)

`PredictionFeaturesSchema` and `StudentRow` fix the canonical feature set at the I/O boundary. Anything outside that list cannot physically reach a model.

### 3.2 Column-transformer-level engineering

A single sklearn `ColumnTransformer` (defined in `features.py`) declares:

- numeric columns → `StandardScaler`;
- ordinal categorical → `OrdinalEncoder(handle_unknown="use_encoded_value")`;
- engineered **academic-momentum** features (approval / enrollment ratios, grade deltas across the two semesters) computed by a bespoke `MomentumTransformer`.

This transformer is part of the `Pipeline`, so it is fit once on train and re-applied bit-for-bit at inference — no train/serve skew.

### 3.3 Model-driven feature importance

Once the five models are trained, the pipeline computes for the XGBoost candidate:

- `booster.get_score(importance_type="gain")` — used to rank all features by the marginal objective-function improvement they contribute at split time.
- sklearn `permutation_importance` on the validation split — used to sanity-check the gain ranking and detect features whose apparent importance is an artefact of column correlation.

The resulting ranking is written to `reports/feature_importance.json` and surfaced on the admin dashboard. The **ablation pass** (`backend/app/ml/ablation.py`) then retrains XGBoost with each feature family removed in turn (academic, demographic, macro-economic, financial-aid) and records the macro-F1 drop in `reports/ablation.md`. Feature families whose removal drops macro-F1 by less than 0.5 pp are candidates for pruning at the next retrain.

**Net effect:** selection is data-driven (gain + permutation + ablation), pipeline-internal (fit once, applied everywhere), and reviewable (one Markdown report per retrain).

---

## 4. Automated retraining

Retraining is a **closed loop** driven by three possible triggers:

| Trigger kind | Source                                    | Gate                                     |
|--------------|-------------------------------------------|------------------------------------------|
| Drift        | `/monitoring/drift` → share ≥ τ (0.30)    | compare_for_promotion + HITL if enabled  |
| Scheduled    | cron (nightly / weekly)                   | same                                     |
| Manual       | `POST /api/v1/retrain` (JWT admin)        | same                                     |

### 4.1 Drift detection

[backend/app/monitoring/drift.py](../backend/app/monitoring/drift.py) builds an Evidently `DataDriftPreset` against the frozen reference parquet and returns a `DriftResult`:

```python
DriftResult(drift_share, n_drifted, n_total, report_path, detected)
```

`detected = share >= threshold` (default 0.30, overridable per request). The HTML report is archived to `reports/drift/drift_<timestamp>.html` and embedded in the Angular dashboard via an `<iframe>`. The share value is published to Prometheus as a gauge so Grafana can alert on sustained drift.

### 4.2 Challenger training

[backend/app/monitoring/retraining.py:run_retraining](../backend/app/monitoring/retraining.py) spawns a fresh subprocess running the exact DVC training command:

```bash
python -m backend.app.ml.train run \
  --train data/processed/train.parquet \
  --val   data/processed/val.parquet \
  --models-out models/staging
```

stdout is streamed line-by-line to the dashboard via a Server-Sent-Events endpoint so the reviewer watches the training log in real time. The new model is written to `models/staging/model.joblib` — *never* overwriting the live champion.

### 4.3 The promotion gate

[backend/app/ml/registry.py:compare_for_promotion](../backend/app/ml/registry.py) enforces three cumulative gates:

1. **Effect-size gate.**  `macro_F1(challenger) − macro_F1(champion) ≥ 0.01` (≥ +1.0 pp).
2. **No-class-regression gate.**  For every class `c`, `F1_c(challenger) − F1_c(champion) ≥ −0.02`.
3. **Statistical-significance gate.**  On paired holdout predictions, McNemar's mid-p test must reject the null of equal error rates at α = 0.05 (the paired version uses the discordant pair counts `b`, `c`).

If *any* gate fails the challenger is **rejected** — atomically, the live model is not touched.

### 4.4 Promotion logic

When all gates pass **and** `AUTO_PROMOTE_ENABLED=true`:

1. `shutil.copy2(staging_model, champion.joblib.new)` then `.replace(champion.joblib)` — an **atomic** POSIX rename, safe across crashes.
2. `metadata.json` mirrored.
3. MLflow `register_and_promote` creates a new registered-model version and transitions it to `Production`, archiving the previous.
4. Prometheus `MODEL_MACRO_F1` gauge is refreshed and `RETRAIN_TOTAL{outcome="promoted"}` is incremented.
5. An `AuditEntry` is appended to `reports/retraining/history.jsonl` with timestamp, trigger, both macro-F1s, per-class deltas, McNemar (p, b, c), n_test, and the new registered version.

When `AUTO_PROMOTE_ENABLED=false`, steps 1–4 are deferred; the challenger sits in the **Staging** stage of the MLflow registry and the dashboard surfaces an `Approve` button.

---

## 5. Human-in-the-Loop (HITL) override

### 5.1 The configuration flag

`AUTO_PROMOTE_ENABLED` lives in `.env`, surfaces in `backend/app/core/config.py:Settings`, and is hot-readable at every retrain invocation — no restart required.

### 5.2 The review UI

The Angular admin route `/admin/model-registry` shows:

- the current production and staging versions side-by-side;
- the frozen holdout **leaderboard** and **confusion matrices**;
- **per-class F1 deltas** colour-coded red / green;
- the **McNemar p-value** and contingency (`b`, `c`);
- the **fairness audit** for every sensitive attribute, with a red flag when any equal-opportunity gap exceeds 0.10;
- a SHAP summary plot for the challenger;
- the **Approve** / **Reject** buttons, disabled unless the reviewer holds an `admin` JWT role.

### 5.3 The approval API

`POST /api/v1/model/registry/approve` (JWT, role=admin):

1. Re-runs the promotion gate with the *current* champion metrics (defensive: the champion may have been replaced by an auto-promotion concurrently).
2. Executes steps 1–5 of §4.4 above.
3. Appends a second audit entry with `trigger="hitl:approve:<reviewer_username>"`.

`POST /api/v1/model/registry/reject` is the mirror: it purges the staging bundle and writes an audit entry with the reviewer's free-text reason.

### 5.4 Why the flag exists

The flag gives the institution a single point of control over autonomy. Early in deployment (low evidence, low trust) HITL is `ON`; as the audit trail accumulates and the reviewer's decisions converge with the gate's recommendations, the institution can flip it off. The flag itself is logged on every retrain, so the governance record reflects the prevailing policy at any past moment.

---

## 6. FastAPI endpoint inventory

| Method | Path                                   | Auth          | Purpose                                                  |
|--------|----------------------------------------|---------------|----------------------------------------------------------|
| POST   | `/api/v1/predict`                      | API key       | single-student risk + top-k SHAP + intervention list     |
| POST   | `/api/v1/predict/batch`                | API key       | CSV upload → risk-scored CSV + summary                   |
| GET    | `/api/v1/monitoring/drift`             | API key       | live drift share + HTML report link                      |
| POST   | `/api/v1/retrain`                      | JWT admin     | trigger challenger (streams training log)                |
| POST   | `/api/v1/model/registry/approve`       | JWT admin     | HITL approve staging → production                        |
| POST   | `/api/v1/model/registry/reject`        | JWT admin     | HITL reject staging; archive with reason                 |
| GET    | `/api/v1/model/registry`               | API key       | production + staging metadata                            |
| GET    | `/api/v1/students/cohort`              | API key       | cached cohort scores for dashboard                       |
| GET    | `/metrics`                             | none          | Prometheus scrape (latency, volume, macro-F1, retrains)  |

Pandera validates every inbound payload at the schema layer — a malformed request returns `422` before any model code runs.

---

## 7. Observability

- **Prometheus** (`infrastructure/prometheus.yml`) scrapes `/metrics` every 15s.
- **Custom metrics** (`backend/app/core/metrics.py`):
  - `request_latency_seconds_histogram{route,status}`;
  - `predictions_total_counter{prediction}`;
  - `MODEL_MACRO_F1` gauge (refreshed on every retrain);
  - `RETRAIN_TOTAL{outcome=promoted|rejected|failed}`;
  - `DRIFT_SHARE_GAUGE`.
- **Grafana** (`infrastructure/grafana/`) ships four provisioned dashboards: *API Health*, *Prediction Volume*, *Model Health*, *Drift & Retraining*.
- **Audit logs** live in `reports/retraining/history.jsonl` (promotion decisions) and `reports/drift/drift_*.html` (drift snapshots); both are git-tracked.

---

## 8. Security posture

- Public endpoints behind an **API key** (`X-API-Key`), rotated via `.env`.
- Admin endpoints behind **JWT** with a short TTL (`JWT_SECRET`, configurable TTL).
- Rate-limiting via `slowapi` on `/predict` and `/predict/batch`.
- CORS is whitelist-based (not `*`) and driven from `.env`.
- Secrets are never committed; `.env.example` is the only file in git.
- Docker images run as a non-root user; `read_only: true` on the app volume in production compose.

---

## 9. Reproducibility checklist

- [x] `dvc repro` rebuilds the full pipeline from a clean clone.
- [x] `docker compose up --build` brings the whole stack up on one command.
- [x] All random seeds fixed in `params.yaml` (`seed: 42`).
- [x] All dependencies pinned in `pyproject.toml`; lockfile committed.
- [x] CI (`.github/workflows/ci.yml`) runs ruff + mypy + pytest with **≥ 80% coverage**.
- [x] Container images published to GHCR on git tag.
- [x] Every experiment has a run ID in MLflow (`mlruns/`).
- [x] Every promotion writes a JSONL audit line.
- [x] Every drift check writes an HTML report.

---

## 10. File-location cheat sheet

| Concern                         | Path                                                              |
|---------------------------------|-------------------------------------------------------------------|
| Pipeline DAG                    | [dvc.yaml](../dvc.yaml)                                           |
| Hyperparameters & gates         | [params.yaml](../params.yaml)                                     |
| Schemas (Pandera)               | [backend/app/ml/schemas.py](../backend/app/ml/schemas.py)         |
| Feature engineering             | [backend/app/ml/features.py](../backend/app/ml/features.py)       |
| Five-model trainer              | [backend/app/ml/train.py](../backend/app/ml/train.py)             |
| Evaluation (bootstrap, fairness)| [backend/app/ml/evaluate.py](../backend/app/ml/evaluate.py)       |
| Ablation                        | [backend/app/ml/ablation.py](../backend/app/ml/ablation.py)       |
| Promotion gate (3 rules)        | [backend/app/ml/registry.py](../backend/app/ml/registry.py)       |
| Drift                           | [backend/app/monitoring/drift.py](../backend/app/monitoring/drift.py) |
| Retraining loop                 | [backend/app/monitoring/retraining.py](../backend/app/monitoring/retraining.py) |
| Audit trail                     | [reports/retraining/history.jsonl](../reports/retraining/history.jsonl) |
| Champion metrics                | [models/champion/metadata.json](../models/champion/metadata.json) |
| Full evaluation dump            | [reports/evaluation.json](../reports/evaluation.json)             |
| Prometheus config               | [infrastructure/prometheus.yml](../infrastructure/prometheus.yml) |
| Compose file                    | [docker-compose.yml](../docker-compose.yml)                       |
