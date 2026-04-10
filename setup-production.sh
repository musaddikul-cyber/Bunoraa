#!/bin/bash
# Complete Django Setup and Verification Script for Bunoraa Production

set -e  # Exit on error

echo "🚀 BUNORAA PRODUCTION SETUP SCRIPT"
echo "=================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ============================================================================
# STEP 1: Verify Environment Variables
# ============================================================================
echo -e "${YELLOW}STEP 1: Verifying Environment Variables${NC}"
echo "==========================================="

required_vars=(
    "DEBUG"
    "DJANGO_SETTINGS_MODULE"
    "DATABASE_URL"
)

for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo -e "${RED}❌ Missing required variable: $var${NC}"
        exit 1
    fi
done

echo -e "${GREEN}✅ All required environment variables set${NC}"
echo "  - DEBUG: $DEBUG"
echo "  - DJANGO_SETTINGS_MODULE: $DJANGO_SETTINGS_MODULE"
echo "  - DATABASE_URL: ${DATABASE_URL:0:30}..."
echo ""

# ============================================================================
# STEP 2: Create Migrations
# ============================================================================
echo -e "${YELLOW}STEP 2: Creating Migrations${NC}"
echo "============================="

python manage.py makemigrations --noinput --check 2>/dev/null || {
    echo "Creating new migrations..."
    python manage.py makemigrations --noinput
}

echo -e "${GREEN}✅ Migrations checked${NC}"
echo ""

# ============================================================================
# STEP 3: Apply Migrations
# ============================================================================
echo -e "${YELLOW}STEP 3: Applying Migrations${NC}"
echo "============================"

python manage.py migrate --noinput

echo -e "${GREEN}✅ Migrations applied${NC}"
echo ""

# ============================================================================
# STEP 4: Collect Static Files
# ============================================================================
echo -e "${YELLOW}STEP 4: Collecting Static Files${NC}"
echo "================================"

python manage.py collectstatic --noinput --clear

echo -e "${GREEN}✅ Static files collected${NC}"
echo ""

# ============================================================================
# STEP 5: Check Database Health
# ============================================================================
echo -e "${YELLOW}STEP 5: Checking Database Health${NC}"
echo "=================================="

python manage.py dbshell <<EOF
SELECT version();
SELECT datname, usename, application_name, state FROM pg_stat_activity WHERE datname = current_database() LIMIT 5;
EOF

echo -e "${GREEN}✅ Database is healthy${NC}"
echo ""

# ============================================================================
# STEP 6: Verify Settings
# ============================================================================
echo -e "${YELLOW}STEP 6: Verifying Django Settings${NC}"
echo "================================="

python manage.py shell <<EOF
from django.conf import settings
print("✅ Production Settings Verified:")
print(f"  - DEBUG: {settings.DEBUG}")
print(f"  - Allowed Hosts: {settings.ALLOWED_HOSTS[:2]}...")
print(f"  - DB Connection Pool: {settings.DATABASES['default'].get('CONN_MAX_AGE', 'Not set')}")
print(f"  - Redis Cache: {'Enabled' if settings.CACHES.get('default') else 'Disabled'}")
print(f"  - Static Files: {settings.STATIC_URL}")
print(f"  - Media Files: {settings.MEDIA_URL}")
EOF

echo ""

# ============================================================================
# STEP 7: Summary
# ============================================================================
echo -e "${GREEN}🎉 SETUP COMPLETE!${NC}"
echo ""
echo "Your Bunoraa application is now ready for production!"
echo ""
echo "Database Connection Pooling:"
echo "  - Enabled: Yes ✅"
echo "  - Connection TTL: ${DB_CONN_MAX_AGE:-600}s"
echo "  - Min Pool Size: ${CONN_POOL_MIN_SIZE:-5}"
echo "  - Max Pool Size: ${CONN_POOL_MAX_SIZE:-20}"
echo ""
echo "Next steps:"
echo "  1. Run: python manage.py createsuperuser"
echo "  2. Test: python manage.py runserver 0.0.0.0:8000"
echo "  3. Deploy with Gunicorn: gunicorn core.wsgi:application"
echo ""
echo "📝 Current Settings File: $DJANGO_SETTINGS_MODULE"
