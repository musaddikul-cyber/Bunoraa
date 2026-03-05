#!/usr/bin/env bash
set -euo pipefail

# Install dependencies already handled by Render
echo "Starting build script..."
pip install -r requirements.txt

is_true() {
  case "${1:-}" in
    1|true|TRUE|True|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

# Install ML dependencies only when ML/chat assistant is enabled.
if is_true "${ML_ENABLED:-0}" || is_true "${ML_CHAT_ASSISTANT:-0}" || is_true "${CHAT_AI_LOCAL_MODEL_ENABLED:-0}"; then
  echo "Installing ML requirements..."
  pip install -r requirements-ml.txt
else
  echo "Skipping ML requirements (ML/chat assistant disabled)"
fi

# Run migrations and collectstatic
# Never generate migrations during deploy. This can create files inside
# site-packages (as seen with admin_interface) and make builds nondeterministic.
python manage.py migrate --noinput
# python manage.py seed_admin_interface_themes || true
# python manage.py sync_admin_interface_theme || true
python manage.py collectstatic --noinput --clear

# Optimize production setup (indexes, analyze, compression)
python manage.py setup_production --optimize-db --create-indexes || true

# Setup periodic tasks and SEO schedules
python manage.py setup_seo_schedules || true

# Skip prerender - consumes too much memory on free tier (512MB limit)
# python manage.py prerender_top --categories=10 --products=20 --include-static || true

echo "Build script completed."
