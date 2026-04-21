#!/usr/bin/env bash
# =============================================================================
# First-time bootstrap for the /opt/uz-dropout deploy directory.
# Run ONCE on a fresh EC2 host, after cloud-init has installed Docker.
#
#   ssh ec2-user@<EC2_HOST>
#   sudo curl -fsSL https://raw.githubusercontent.com/<owner>/<repo>/main/scripts/ec2_bootstrap.sh -o /tmp/bootstrap.sh
#   sudo bash /tmp/bootstrap.sh <owner>/<repo>
#
# Idempotent — re-running is a no-op for directories that already exist.
# =============================================================================
set -euo pipefail

REPO="${1:-}"
if [ -z "$REPO" ]; then
  echo "usage: $0 <github-owner>/<github-repo>" >&2
  exit 1
fi

DEPLOY_DIR="/opt/uz-dropout"
REPO_URL="https://github.com/${REPO}.git"

echo "[bootstrap] ensuring $DEPLOY_DIR exists…"
mkdir -p "$DEPLOY_DIR"
chown ec2-user:ec2-user "$DEPLOY_DIR"

echo "[bootstrap] checking Docker…"
if ! command -v docker >/dev/null 2>&1; then
  echo "[bootstrap] installing Docker…"
  dnf -y install docker
  systemctl enable --now docker
  usermod -aG docker ec2-user
fi

if ! docker compose version >/dev/null 2>&1; then
  echo "[bootstrap] installing Docker Compose v2 plugin…"
  mkdir -p /usr/local/lib/docker/cli-plugins
  curl -fSL "https://github.com/docker/compose/releases/download/v2.29.7/docker-compose-linux-x86_64" \
    -o /usr/local/lib/docker/cli-plugins/docker-compose
  chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
fi

if ! command -v git >/dev/null 2>&1; then
  dnf -y install git
fi

# Clone (or fetch) the repo into a side directory. We only need it for
# DVC stage definitions, Grafana/Prometheus configs, and scripts — the
# deploy workflow rsyncs those into $DEPLOY_DIR, but a first-time manual
# bootstrap before any push needs them locally to run `dvc repro`.
SRC_DIR="/opt/uz-dropout-src"
# /opt is root-owned, so the parent has to be created as root with
# ec2-user ownership *before* the unprivileged git clone.
mkdir -p "$SRC_DIR"
chown -R ec2-user:ec2-user "$SRC_DIR"

if [ ! -d "$SRC_DIR/.git" ]; then
  echo "[bootstrap] cloning $REPO_URL → $SRC_DIR"
  sudo -u ec2-user git clone "$REPO_URL" "$SRC_DIR"
else
  echo "[bootstrap] $SRC_DIR already cloned — fetching latest…"
  sudo -u ec2-user git -C "$SRC_DIR" fetch --all --prune
  sudo -u ec2-user git -C "$SRC_DIR" reset --hard origin/main
fi

echo "[bootstrap] seeding deploy dir from source tree…"
sudo -u ec2-user cp -r \
  "$SRC_DIR/docker-compose.prod.yml" \
  "$SRC_DIR/dvc.yaml" \
  "$SRC_DIR/params.yaml" \
  "$SRC_DIR/infrastructure" \
  "$SRC_DIR/scripts" \
  "$DEPLOY_DIR/"

echo "[bootstrap] creating writable state directories…"
sudo -u ec2-user mkdir -p \
  "$DEPLOY_DIR/models/champion" \
  "$DEPLOY_DIR/data/reference" \
  "$DEPLOY_DIR/data/processed" \
  "$DEPLOY_DIR/reports/drift" \
  "$DEPLOY_DIR/mlruns"

chmod +x "$DEPLOY_DIR/scripts/"*.sh

echo
echo "[bootstrap] done."
echo
echo "Next steps:"
echo "  1. Create $DEPLOY_DIR/.env from the project's .env.prod.example and fill secrets."
echo "  2. Run the first training pass to seed models/ + data/reference/:"
echo "       cd $DEPLOY_DIR && docker compose -f docker-compose.prod.yml --profile train run --rm trainer"
echo "  3. Push to main on GitHub — the deploy workflow will take over from here."
