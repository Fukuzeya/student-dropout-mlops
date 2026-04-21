# Local Setup Guide

End-to-end instructions for running the **UZ Student Dropout MLOps** stack on a single machine.

There are two supported paths:

- **Path A — Docker Compose (recommended)**: brings up API + frontend + MLflow + Postgres + Prometheus + Grafana with one command. Best for demoing the full system.
- **Path B — Native (Python venv + Node)**: faster inner loop for backend / frontend development. Recommended while you are iterating on code.

You can mix the two: e.g. run the database / MLflow stack from Compose and the API from your venv.

---

## 0. Prerequisites

| Tool | Version | Notes |
|---|---|---|
| **Python** | 3.11.x or 3.12.x | `pyproject.toml` pins `>=3.11,<3.13` |
| **Node.js** | 20 LTS | required by Angular 20 |
| **npm** | 10+ | ships with Node 20 |
| **Git** | any recent | |
| **Docker Desktop** | 4.x | only for Path A |
| **Make** *(optional)* | any | nice for shortcuts; not required |

Windows users: a working **PowerShell 7** (`pwsh`) or **Git Bash** is fine. Commands below use bash-style syntax — translate `cp` → `copy`, forward slashes are accepted by Python on Windows.

Optional but useful:

- **`uv`** (fast Python package manager) — `pip install uv` then substitute `uv pip install …`
- **VS Code** with the Python, Pylance, Angular Language Service extensions
- **DBeaver** / **pgAdmin** to peek inside the MLflow Postgres

---

## 1. Clone & bootstrap secrets

```bash
git clone <your-repo-url> student-dropout-mlops
cd student-dropout-mlops

cp .env.example .env
```

Edit `.env` and set at minimum:

```ini
API_KEY=<paste output of: openssl rand -hex 32>
JWT_SECRET=<paste output of: openssl rand -hex 32>
ADMIN_PASSWORD=<your-admin-password>
```

The defaults for `ADMIN_USERNAME=admin`, MLflow URIs, and Postgres creds are fine for local use. **Never commit your filled `.env`** — it's gitignored.

---

## 2. Path A — Docker Compose (recommended for demos)

### 2.1 Start the stack

```bash
docker compose up --build -d
```

First build downloads ~2 GB of base images and may take 5–10 minutes. Subsequent runs are cached.

Once the build is done, the following services are up:

| Service | URL | Credentials |
|---|---|---|
| Frontend (Angular) | http://localhost:4200 | login w/ `ADMIN_USERNAME` / `ADMIN_PASSWORD` |
| FastAPI + Swagger | http://localhost:8000/docs | `X-API-Key: <API_KEY>` for `/predict*` |
| MLflow | http://localhost:5000 | none |
| Prometheus | http://localhost:9090 | none |
| Grafana | http://localhost:3000 | `admin` / `admin` (override via `GRAFANA_PASSWORD`) |

Check health:

```bash
curl http://localhost:8000/api/v1/monitoring/health
docker compose ps   # all should show "healthy" or "running"
```

### 2.2 Train the champion model (one-time)

The API ships with **no model** until you run the DVC pipeline. You have two options:

**Option A — One-shot trainer container (no host Python needed):**

```bash
docker compose --profile train run --rm trainer
```

This spins up a short-lived container that shares the API image (so all deps are already baked in), initialises DVC if needed, runs `dvc repro` (download → validate → preprocess → train → evaluate), and exits. Artefacts land in `./models`, `./data`, `./reports`, `./mlruns` on the host via bind mounts. Takes ~5–15 min on first run, faster after caching.

**Option B — Native venv:** if you already have a Python venv set up (see §3), just run `dvc repro` from the repo root.

After either path, restart the API so it picks up the new champion:

```bash
docker compose restart api
```

You should now see `model_loaded: true` at `/api/v1/monitoring/health`.

### 2.3 Stop the stack

```bash
docker compose down               # stops containers, keeps volumes
docker compose down -v            # also wipes Postgres / MLflow artefacts / Grafana state
```

---

## 3. Path B — Native development

This is the path you'll use while writing code. Hot reload on both backend and frontend.

### 3.1 Backend (Python venv)

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

That installs every runtime dep (FastAPI, scikit-learn, XGBoost, LightGBM, MLflow, Evidently, SHAP, DVC, …) plus dev tools (ruff, mypy, pytest, httpx).

> **Windows + LightGBM gotcha:** `pip install lightgbm` sometimes installs a wheel without the bundled DLL. If `python -c "import lightgbm"` raises `Cannot find lightgbm library file`, reinstall with `pip install --force-reinstall --no-binary :all: lightgbm` (requires a C++ build toolchain) **or** run the affected tests via Docker. The XGBoost-only training path is unaffected.

### 3.2 Train + register the champion

```bash
dvc repro
```

This runs the full lineage: `download → validate → preprocess → train → evaluate`. You'll get:

- `data/raw/dropout.csv` — UCI archive
- `data/processed/{train,val,test}.parquet` — stratified splits
- `data/reference/reference.parquet` — the snapshot drift checks compare against
- `models/champion/model.joblib` — promoted bundle
- `reports/evaluation.json` — bootstrap CIs, calibration, threshold, cost, fairness
- `reports/figures/calibration_{pre,post}.png` — reliability diagrams
- MLflow runs under `mlruns/` (or your remote MLflow server, if `MLFLOW_TRACKING_URI` is overridden)

Inspect runs:

```bash
mlflow ui --port 5000   # open http://localhost:5000
```

### 3.3 Run the API

```bash
# from repo root, with venv active
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

- Swagger: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Prometheus metrics: http://localhost:8000/metrics

Smoke-test a prediction:

```bash
# get a JWT (admin endpoints)
curl -s -X POST http://localhost:8000/api/v1/auth/token \
  -d "username=admin&password=$ADMIN_PASSWORD" | jq -r .access_token
```

```bash
# single prediction (API key)
curl -X POST http://localhost:8000/api/v1/predict \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d @backend/tests/fixtures/sample_student.json
```

### 3.4 Run the frontend

```bash
cd frontend
npm install            # first time only, ~2-3 min
npm start              # ng serve on :4200, hot reload
```

Open http://localhost:4200. The dev environment file points at `http://localhost:8000/api/v1`, so it talks to your local API.

**First-time auth setup in the UI:**
1. Click the API key field in the top bar → paste the value of `API_KEY` from your `.env`. This unlocks `/predict*`.
2. Use the **Login** page with `ADMIN_USERNAME` / `ADMIN_PASSWORD` to get a JWT. This unlocks Admin (manual retrain, registry promotion) and the **drift-driven auto-retrain** panel on Monitoring.

---

## 4. Verify everything works

```bash
# Backend tests (51 should pass; 3 may skip locally — see Troubleshooting)
pytest --no-cov

# Lint + type check
ruff check .
mypy backend/app

# Frontend build (catches template errors)
cd frontend && npm run build
```

End-to-end happy path in the UI:

1. **Dashboard** — KPI tiles populate from `/monitoring/health`, `/monitoring/kpis`.
2. **Students** — paginated list with risk chips.
3. **Student detail** — open a student → SHAP waterfall + intervention card.
4. **Batch predict** — drop a CSV → results table.
5. **Monitoring** — see evaluation rigor card (CIs, calibration, fairness), drift status, and the *Drift-driven auto-retrain* panel (admin only).
6. **Admin** — manual retrain → audit log row appears with McNemar p-value.

---

## 5. Common operations

### Re-run a single DVC stage

```bash
dvc repro evaluate           # only the evaluation step
dvc repro --force train      # rerun training even if inputs are unchanged
```

### Reset trained artefacts

```bash
rm -rf models/champion/* reports/figures/* reports/evaluation.json
dvc repro
```

### Tail API logs in Docker

```bash
docker compose logs -f api
```

### Drop into the API container

```bash
docker compose exec api bash
```

### Trigger drift-driven auto-retrain from CLI

```bash
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/token \
  -d "username=admin&password=$ADMIN_PASSWORD" | jq -r .access_token)

curl -X POST "http://localhost:8000/api/v1/monitoring/drift/auto-retrain?threshold=0.30&force=false" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@path/to/production-batch.csv"
```

---

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `Cannot find lightgbm library file` on Windows | See §3.1 note. Tests `test_models.py` and `test_api_predict.py` collect-fail without it. |
| `ModuleNotFoundError: No module named 'evidently'` in `test_drift.py` | `pip install evidently>=0.4.33` — included in main deps; reinstall with `pip install -e .` if missing. |
| `Reference snapshot missing` on `/monitoring/drift*` | Run `dvc repro preprocess`. The reference parquet is produced by that stage. |
| `model_loaded: false` at `/monitoring/health` | Run `dvc repro` (or just `dvc repro train`) and restart the API. Compose: `docker compose restart api`. |
| Frontend shows 401 on every call | Set the API key in the top-bar field, then log in for admin views. JWT lives in `localStorage` under `uz-ews.auth`. |
| Compose `api` keeps restarting with `API_KEY is required` | You forgot to copy `.env.example` → `.env`, or you started Compose from a directory without the `.env`. Compose loads `.env` from the current working dir. |
| MLflow UI shows no runs | DVC writes to the local `mlruns/` by default. Either run `mlflow ui` from the repo root, or set `MLFLOW_TRACKING_URI=http://localhost:5000` and re-run `dvc repro train`. |
| Port already in use (`:8000`, `:4200`, `:5000`, `:3000`, `:9090`) | Kill the offending process or change the host port in `docker-compose.yml` / your `uvicorn` / `ng serve` command. |
| `dvc repro` complains about `dvc.lock` mismatch | `dvc repro --force` rebuilds; commit the new lockfile. |

---

## 7. Cleanup

```bash
docker compose down -v           # remove containers + named volumes
deactivate                       # exit the venv
rm -rf .venv mlruns reports models/champion/* data/raw data/processed data/reference
```

That returns the repo to a fresh-clone state.
