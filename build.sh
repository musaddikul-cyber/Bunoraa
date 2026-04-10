#!/usr/bin/env bash
set -euo pipefail

# Install backend dependencies
echo "Starting build script..."
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_ROOT_USER_ACTION=ignore

pip install --prefer-binary -r requirements.txt

is_true() {
  case "${1:-}" in
    1|true|TRUE|True|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

# Install ML dependencies only when ML/chat assistant is enabled.
if is_true "${ML_ENABLED:-0}" || is_true "${ML_CHAT_ASSISTANT:-0}" || is_true "${CHAT_AI_LOCAL_MODEL_ENABLED:-0}"; then
  echo "Installing ML requirements..."
  pip install --prefer-binary -r requirements-ml.txt
else
  echo "Skipping ML requirements (ML/chat assistant disabled)"
fi

process_type="${PROCESS_TYPE:-web}"
if [ "${process_type}" = "web" ]; then
  # Run migrations and collectstatic
  # Never generate migrations during deploy. This can create files inside
  # site-packages (as seen with admin_interface) and make builds nondeterministic.
  # Skip framework checks here to avoid third-party admin_interface false-positive
  # migration drift under S3 storage backends.
  python manage.py migrate --noinput --skip-checks
  # python manage.py seed_admin_interface_themes || true
  # python manage.py sync_admin_interface_theme || true
  python manage.py collectstatic --noinput

  # Optional: expensive one-time optimizer tasks.
  if is_true "${RUN_PRODUCTION_OPTIMIZERS:-0}"; then
    python manage.py setup_production --optimize-db --create-indexes || true
  else
    echo "Skipping production optimizer tasks (set RUN_PRODUCTION_OPTIMIZERS=true to enable)"
  fi

  # Optional: setup/update SEO schedules during deploy.
  if is_true "${RUN_SEO_SCHEDULE_SETUP:-0}"; then
    python manage.py setup_seo_schedules || true
  else
    echo "Skipping SEO schedule setup (set RUN_SEO_SCHEDULE_SETUP=true to enable)"
  fi
else
  echo "Skipping migrations/collectstatic (PROCESS_TYPE=${process_type})"
fi

# Skip prerender - consumes too much memory on free tier (512MB limit)
# python manage.py prerender_top --categories=10 --products=20 --include-static || true

echo "Build script completed."
