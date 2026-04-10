#!/bin/bash
# Startup Performance Test & Optimization Script

echo "🚀 Django Startup Performance Test"
echo "==================================="
echo ""

# Set optimization environment variables
export SKIP_MIGRATIONS_CHECK=True
export ML_ENABLED=False
export PROCESS_TYPE=web
export GUNICORN_PRELOAD=True
export GUNICORN_WORKER_CLASS=sync
export GUNICORN_LOG_LEVEL=warning

echo "Environment Optimizations Set:"
echo "  ✅ SKIP_MIGRATIONS_CHECK=True"
echo "  ✅ ML_ENABLED=False"
echo "  ✅ PROCESS_TYPE=web"
echo "  ✅ GUNICORN_PRELOAD=True"
echo "  ✅ GUNICORN_WORKER_CLASS=sync"
echo "  ✅ GUNICORN_LOG_LEVEL=warning"
echo ""

# Test 1: Django System Check
echo "📊 Test 1: Django System Check"
echo "  Command: python manage.py check"
start_time=$(date +%s%N)
python manage.py check 2>&1 | grep -E "System check|issues"
end_time=$(date +%s%N)
check_time=$((($end_time - $start_time) / 1000000))
echo "  ⏱️  Time: ${check_time}ms"
echo ""

# Test 2: Test Database Connection
echo "📊 Test 2: Database Connection"
echo "  Command: python manage.py dbshell (exit)"
start_time=$(date +%s%N)
echo "exit\q" | python manage.py dbshell 2>&1 > /dev/null
end_time=$(date +%s%N)
db_time=$((($end_time - $start_time) / 1000000))
echo "  ⏱️  Time: ${db_time}ms"
echo ""

# Test 3: Cache Connection
echo "📊 Test 3: Redis Cache Connection"
python manage.py shell <<EOF 2>&1 | head -5
from django.core.cache import cache
import time
start = time.time()
cache.set('test', 'value', 60)
result = cache.get('test')
elapsed = (time.time() - start) * 1000
print(f'Cache test: {elapsed:.0f}ms')
print(f'Value: {result}')
EOF
echo ""

# Test 4: Gunicorn Startup
echo "📊 Test 4: Gunicorn Startup Time (3 seconds)"
echo "  Command: gunicorn --config gunicorn_conf.py core.wsgi:application (timeout 3s)"
timeout 3 gunicorn \
  --config gunicorn_conf.py \
  --workers=1 \
  --bind=127.0.0.1:8001 \
  core.wsgi:application \
  2>&1 | head -20 &
sleep 1
curl -s http://127.0.0.1:8001/admin/ > /dev/null 2>&1 && echo "  ✅ Server responded in ~1 second" || echo "  ⏳ Server not ready yet"
echo ""

# Performance Summary
echo "🎯 Performance Summary"
echo "====================="
if [ $check_time -lt 5000 ]; then
  echo "✅ Django check: ${check_time}ms (EXCELLENT - < 5s)"
elif [ $check_time -lt 10000 ]; then
  echo "🟡 Django check: ${check_time}ms (GOOD - < 10s)"
else
  echo "🔴 Django check: ${check_time}ms (SLOW - > 10s)"
fi

echo ""
echo "💡 To optimize further:"
echo "  1. Set SKIP_MIGRATIONS_CHECK=True in .env"
echo "  2. Set ML_ENABLED=False on web servers"
echo "  3. Use GUNICORN_PRELOAD=True"
echo "  4. Change GUNICORN_WORKER_CLASS to 'sync'"
echo "  5. Reduce REDIS_SOCKET_CONNECT_TIMEOUT to 2s"
echo ""
echo "📝 See DJANGO_STARTUP_OPTIMIZATION.md for more tips"
