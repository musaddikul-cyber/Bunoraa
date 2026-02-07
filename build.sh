#!/usr/bin/env bash
set -euo pipefail

# Install dependencies already handled by Render
echo "Starting build script..."
pip install -r requirements.txt
# Skip ML requirements - disabled for memory optimization on Render free tier
# pip install -r requirements-ml.txt

# Run migrations and collectstatic
python manage.py makemigrations --noinput
python manage.py migrate --noinput
python manage.py collectstatic --noinput

# Optimize production setup (indexes, analyze, compression)
python manage.py setup_production --optimize-db --create-indexes || true

# Setup periodic tasks and SEO schedules
python manage.py setup_seo_schedules || true

# Skip prerender - consumes too much memory on free tier (512MB limit)
# python manage.py prerender_top --categories=10 --products=20 --include-static || true

echo "Build script completed."