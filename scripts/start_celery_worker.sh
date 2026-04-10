#!/usr/bin/env bash
set -euo pipefail

: "${CELERY_APP:=core}"
: "${CELERY_LOGLEVEL:=warning}"
: "${CELERY_POOL:=solo}"
: "${CELERY_CONCURRENCY:=1}"
: "${CELERY_QUEUES:=celery,payments,notifications,chat,analytics,backups}"
: "${CELERY_MAX_TASKS_PER_CHILD:=500}"

exec celery -A "${CELERY_APP}" worker \
  -n "worker@%h" \
  --pool="${CELERY_POOL}" \
  --concurrency="${CELERY_CONCURRENCY}" \
  -Q "${CELERY_QUEUES}" \
  -l "${CELERY_LOGLEVEL}" \
  --max-tasks-per-child="${CELERY_MAX_TASKS_PER_CHILD}"
