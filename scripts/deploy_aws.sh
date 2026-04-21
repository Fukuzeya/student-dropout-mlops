#!/usr/bin/env bash
# =============================================================================
# Called by the GitHub Actions deploy job on the EC2 host. Merges the
# per-deploy image-tag env produced by the workflow with the persistent
# .env, pulls the new images, rolls the stack, and prunes dangling images.
#
# Safe to run manually too:
#   cd /opt/uz-dropout && ./scripts/deploy_aws.sh
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

COMPOSE="docker compose -f docker-compose.prod.yml --env-file .env"
if [ -f .env.deploy ]; then
  COMPOSE="$COMPOSE --env-file .env.deploy"
fi

echo "[deploy] pulling images…"
$COMPOSE pull

echo "[deploy] starting services (detached)…"
$COMPOSE up -d --remove-orphans

echo "[deploy] pruning dangling images…"
docker image prune -f --filter "until=24h" >/dev/null

echo "[deploy] running services:"
$COMPOSE ps

echo "[deploy] done."
