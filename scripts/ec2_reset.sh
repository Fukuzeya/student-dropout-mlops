#!/usr/bin/env bash
# =============================================================================
# Nuclear reset for /opt/uz-dropout. Use when a deployment has wedged and
# you want to start from a clean slate.
#
# WHAT IS PRESERVED
#   * MLflow backend DB + artefacts (mlflow-db, mlflow-artifacts volumes)
#   * Prometheus history (prometheus-data)
#   * Grafana dashboards (grafana-data)
#   Pass --wipe-state to blow those away too.
#
# WHAT IS REMOVED
#   * All running/stopped containers in the compose project
#   * /opt/uz-dropout/models, /opt/uz-dropout/reports, /opt/uz-dropout/mlruns
#   * Docker dangling images
#
# Run as ec2-user (needs docker group membership).
# =============================================================================
set -euo pipefail

WIPE_STATE=0
if [ "${1:-}" = "--wipe-state" ]; then
  WIPE_STATE=1
fi

cd /opt/uz-dropout

echo "[reset] stopping compose stack…"
docker compose -f docker-compose.prod.yml down --remove-orphans

if [ "$WIPE_STATE" -eq 1 ]; then
  echo "[reset] wiping persistent volumes (mlflow/prometheus/grafana)…"
  docker compose -f docker-compose.prod.yml down -v
fi

echo "[reset] clearing model + report + mlruns bind-mounts…"
rm -rf models/champion models/challengers reports/drift/* mlruns/* 2>/dev/null || true

echo "[reset] pruning dangling images…"
docker image prune -f >/dev/null

echo "[reset] done. Start back up with:"
echo "  ./scripts/deploy_aws.sh"
