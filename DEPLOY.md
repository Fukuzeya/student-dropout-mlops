# Deploying the Student Dropout MLOps stack to AWS

This is the overnight-ready playbook. If you follow the steps in order,
every push to `main` from here on will auto-deploy the stack to your
existing EC2 host at `54.156.43.140` via GitHub Actions.

> **Current state.** An EC2 instance, Elastic IP, IAM role, security group,
> S3 bucket, and key pair already exist (see `terraform.tfstate`). You do
> **not** need to re-provision them — the Terraform sources in
> `infrastructure/terraform/` describe that same infra so any future
> rebuild stays reproducible.

---

## 0. Prerequisites on your laptop

| Tool | Version | Purpose |
|---|---|---|
| Git | any | push the repo |
| GitHub CLI (optional) | ≥ 2.40 | set Actions secrets from the shell |
| SSH client | OpenSSH | first-time login to the EC2 host |
| Terraform (optional) | ≥ 1.6 | only needed if you want to re-apply infra |

The matching SSH private key was generated when you created the key pair.
It should live at `~/.ssh/uz-dropout-key` with the public half at
`~/.ssh/uz-dropout-key.pub`. Verify:

```bash
ssh -i ~/.ssh/uz-dropout-key ec2-user@54.156.43.140 'hostname'
```

If that prints a hostname, SSH is working.

---

## 1. Push the repo to GitHub

```bash
cd "c:/Users/hp/Desktop/UZ Masters Resources/student-dropout-mlops"

git init -b main
git add .
git commit -m "feat: initial commit — Level-5 MLOps stack for UZ dropout project"

# Replace with your own GitHub repo URL.
git remote add origin https://github.com/<your-username>/student-dropout-mlops.git
git push -u origin main
```

> The `terraform.tfstate` and `terraform.tfvars` files are sensitive.
> Make sure they're covered by `.gitignore` **before** the first commit.
> If they've already been pushed, rotate the SSH key and the AWS access
> key after the migration.

Verify nothing secret landed in the initial commit:

```bash
git ls-files | grep -E 'terraform\.tfstate|terraform\.tfvars|\.env$'
# Should print nothing.
```

---

## 2. Configure GitHub Actions secrets

The deploy workflow at `.github/workflows/deploy.yml` reads the following
secrets. Set them from the GitHub UI (Settings → Secrets and variables →
Actions → New repository secret) or via the CLI:

```bash
gh secret set EC2_HOST --body "54.156.43.140"
gh secret set EC2_SSH_KEY < ~/.ssh/uz-dropout-key    # the PRIVATE key
gh secret set GHCR_PAT   --body "<classic PAT, scope: read:packages>"
```

| Secret | Purpose | Required? |
|---|---|---|
| `EC2_HOST` | Public IP / DNS of the EC2 host | ✅ |
| `EC2_SSH_KEY` | OpenSSH **private** key (whole file, including BEGIN/END) | ✅ |
| `GHCR_PAT` | GitHub classic PAT with `read:packages`. Only needed if the repo is private, because the EC2 host needs to pull `ghcr.io/...` images. | Conditional |

> **Tip.** If you prefer the repo be public, you can skip `GHCR_PAT`
> entirely — GHCR images of public repos pull anonymously.

Also create a GitHub **Environment** named `production` (Settings →
Environments → New environment → `production`) so the deploy job can
attach a public URL and, optionally, a manual approval gate.

---

## 3. First-time EC2 bootstrap

This only needs to happen **once** on the box. Pick the matching repo
slug (`<owner>/<repo>`) for your GitHub account.

```bash
ssh -i ~/.ssh/uz-dropout-key ec2-user@54.156.43.140

# on the EC2 host:
curl -fsSL https://raw.githubusercontent.com/<your-username>/student-dropout-mlops/main/scripts/ec2_bootstrap.sh -o /tmp/bootstrap.sh
sudo bash /tmp/bootstrap.sh <your-username>/student-dropout-mlops
```

The script:

1. Installs Docker + Compose v2 if missing.
2. Clones the repo to `/opt/uz-dropout-src` and seeds `/opt/uz-dropout`
   with `docker-compose.prod.yml`, `dvc.yaml`, `params.yaml`,
   `infrastructure/`, and `scripts/`.
3. Creates the bind-mount directories (`models/`, `data/reference/`,
   `reports/drift/`, `mlruns/`).

---

## 4. Seed the production `.env`

Still on the EC2 host:

```bash
cd /opt/uz-dropout
cp /opt/uz-dropout-src/.env.prod.example .env
nano .env    # or vim
```

Fill in **all** the `replace-with-...` placeholders. Generate strong
values for `API_KEY` and `JWT_SECRET` like so:

```bash
openssl rand -hex 32   # run twice, paste one into each slot
```

Double-check:

* `GHCR_REPOSITORY=<your-username>/student-dropout-mlops` (lowercase).
* `CORS_ALLOW_ORIGINS=http://54.156.43.140:4200`.
* `ADMIN_PASSWORD` is something only you know.

Lock the file down:

```bash
chmod 600 .env
```

---

## 5. First training run

The API won't start without `models/champion/model.joblib` and
`data/reference/reference.parquet`. The `trainer` profile of the compose
file produces both via `dvc repro`.

```bash
cd /opt/uz-dropout
docker compose -f docker-compose.prod.yml --profile train run --rm trainer
```

Expect 5–15 minutes the first time. When it finishes you should see
`models/champion/model.joblib` and a metrics table in the log.

---

## 6. Trigger the first auto-deploy

Make any small change on your laptop and push to `main` — for example:

```bash
git commit --allow-empty -m "chore: trigger first production deploy"
git push origin main
```

Watch the workflow at `https://github.com/<you>/<repo>/actions`. The three
jobs run in order:

1. `build & push api` — builds `backend/Dockerfile`, pushes to
   `ghcr.io/<you>/<repo>/api` with tags `latest` and `sha-<short>`.
2. `build & push frontend` — same for `frontend/Dockerfile`.
3. `roll stack on EC2` — rsyncs the deploy bundle, pins the image tag
   to this commit's SHA, and runs `./scripts/deploy_aws.sh` over SSH.

The workflow finishes with a 100-second health check hitting
`http://54.156.43.140:8000/api/v1/monitoring/health`.

---

## 7. Verify the stack is live

Open each URL in a browser:

| Service | URL |
|---|---|
| Angular dashboard | http://54.156.43.140:4200 |
| FastAPI docs | http://54.156.43.140:8000/docs |
| MLflow (Experiments + **Models registry**) | http://54.156.43.140:5000 |
| Grafana | http://54.156.43.140:3000 |
| Prometheus | http://54.156.43.140:9090 |

Smoke-test the API from your laptop:

```bash
curl -fsSL http://54.156.43.140:8000/api/v1/monitoring/health
# {"status":"ok",...}
```

Log into the dashboard as `admin` / `<ADMIN_PASSWORD you set>` and
confirm the Monitoring tab shows a drift score, the Retrain tab lists
the champion run, and MLflow's **Models** tab shows
`student-dropout-classifier` with at least one Production version.

---

## 8. Day-to-day operations

### Push new code
Just `git push origin main`. The deploy workflow handles the rest.

### Roll back
From your laptop:

```bash
# Find the last good SHA
git log --oneline

# Re-trigger deploy.yml on that commit
gh workflow run deploy.yml --ref <that-commit-sha>
```

Alternatively, pin `IMAGE_TAG` in `/opt/uz-dropout/.env` to the previous
`sha-xxxxxx` and run `./scripts/deploy_aws.sh` on the host.

### Trigger a retrain from the UI
Log into the dashboard → **Retrain** → *Run retrain*. The run streams
progress via SSE. When it promotes a new champion, MLflow's Models tab
automatically gets a new version transitioned to Production with the old
version archived (see `backend/app/ml/registry.py::register_and_promote`).

### Run drift + auto-retrain from the UI
Upload a new batch CSV on the **Monitoring** tab → *Auto-retrain on
drift*. The job runs asynchronously with live logs. If drift is below
the threshold it short-circuits; otherwise it trains challengers and
only promotes when the champion-vs-challenger gate passes.

### SSH into the host for debugging

```bash
ssh -i ~/.ssh/uz-dropout-key ec2-user@54.156.43.140
cd /opt/uz-dropout
docker compose -f docker-compose.prod.yml logs -f api
docker compose -f docker-compose.prod.yml ps
```

### Reset the box

```bash
# On the EC2 host
cd /opt/uz-dropout
./scripts/ec2_reset.sh            # preserves MLflow history
./scripts/ec2_reset.sh --wipe-state    # nuclear option
```

---

## 9. Recreating the infra from scratch (future-proofing)

If you ever need to provision a second environment:

```bash
cd infrastructure/terraform
terraform init
terraform plan -var-file=../../terraform.tfvars
terraform apply -var-file=../../terraform.tfvars
```

The sources in this directory reproduce the exact set of resources
currently tracked in `terraform.tfstate`:

* EC2 `t3.xlarge` on Amazon Linux 2023 in `us-east-1` (default VPC)
* Elastic IP + EIP association
* IAM role + instance profile + S3 policy scoped to the artefacts bucket
* S3 bucket with versioning + public-access-block
* Security group exposing ports 22 / 3000 / 4200 / 5000 / 8000 / 9000 / 9001 / 9090
* SSH key pair
* Cloud-init bootstrap via `user_data.sh.tftpl`

To **adopt** the existing running instance into this module (instead of
creating a duplicate), put the current `terraform.tfstate` next to the
`.tf` files and run `terraform plan`. It should report **0 changes**.

---

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `deploy` job fails at *Rsync deploy bundle* | `EC2_SSH_KEY` secret missing/malformed | Re-add the private key (whole file, including `-----BEGIN` / `-----END`) |
| API health check fails after deploy | First-time start, no `models/champion/` | Run step 5 on the host |
| `docker compose pull` 401 on GHCR | Private repo without `GHCR_PAT` | Create a classic PAT with `read:packages` scope and re-add as secret |
| Retrain never reaches MLflow Models tab | `MLFLOW_TRACKING_URI` wrong | Check `/opt/uz-dropout/.env` — must be `http://mlflow:5000` so the api container reaches MLflow by service name |
| Drift page shows "network error" | Shouldn't happen anymore — it now streams via SSE. If it does, check `docker compose logs api` for a traceback in `_run_drift_in_thread` |
| Dashboard loads but API calls 401 | `API_KEY` in `.env` doesn't match `environment.ts` in the frontend image | Rebuild frontend after rotating the key, then redeploy |

---

## Appendix — secret bootstrap (optional, gh CLI)

Paste-and-go block for setting all Actions secrets in one shot. Replace
the placeholders first.

```bash
REPO="<your-username>/student-dropout-mlops"

gh secret set EC2_HOST   --repo "$REPO" --body "54.156.43.140"
gh secret set EC2_SSH_KEY --repo "$REPO" < ~/.ssh/uz-dropout-key

# Only if the repo is private:
gh secret set GHCR_PAT   --repo "$REPO" --body "<paste-classic-PAT>"
```

That's it. Push, watch, and the pipeline takes over.
