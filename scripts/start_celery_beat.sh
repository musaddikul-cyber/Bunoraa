#!/usr/bin/env bash
set -euo pipefail

: "${CELERY_APP:=core}"
: "${CELERY_LOGLEVEL:=warning}"

if [ -n "${CELERY_BEAT_SCHEDULER:-}" ]; then
  exec celery -A "${CELERY_APP}" beat -l "${CELERY_LOGLEVEL}" --scheduler "${CELERY_BEAT_SCHEDULER}"
fi

exec celery -A "${CELERY_APP}" beat -l "${CELERY_LOGLEVEL}"

