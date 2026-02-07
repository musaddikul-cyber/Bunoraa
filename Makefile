# ============================================
# Bunoraa E-Commerce Platform
# Makefile for Common Commands
# ============================================

.PHONY: help install dev run migrate static test lint format clean docker-up docker-down

# Default target
help:
	@echo "Bunoraa E-Commerce Platform - Available Commands"
	@echo "================================================"
	@echo ""
	@echo "Development:"
	@echo "  make install     Install all dependencies"
	@echo "  make dev         Start development server"
	@echo "  make run         Run Django server"
	@echo "  make shell       Open Django shell"
	@echo "  make dbshell     Open database shell"
	@echo ""
	@echo "Database:"
	@echo "  make migrate     Run database migrations"
	@echo "  make makemigrations  Create new migrations"
	@echo "  make resetdb     Reset database (DANGER!)"
	@echo ""
	@echo "Static Files:"
	@echo "  make static      Collect static files"
	@echo "  make css         Build TailwindCSS"
	@echo "  make css-watch   Watch and build TailwindCSS"
	@echo ""
	@echo "Testing:"
	@echo "  make test        Run all tests"
	@echo "  make test-cov    Run tests with coverage"
	@echo "  make lint        Run linters"
	@echo "  make format      Format code"
	@echo ""
	@echo "Celery:"
	@echo "  make celery      Start Celery worker"
	@echo "  make celery-beat Start Celery beat"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-up   Start Docker containers"
	@echo "  make docker-down Stop Docker containers"
	@echo "  make docker-logs View Docker logs"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean       Clean up temporary files"
	@echo "  make superuser   Create superuser"
	@echo "  make messages    Generate translation messages"
	@echo "  make compile     Compile translation messages"

# ============================================
# Development
# ============================================

install:
	pip install -r requirements.txt
	npm install --legacy-peer-deps
	python manage.py migrate
	npm run build:css

dev:
	@echo "Starting development server..."
	@make -j2 run css-watch

run:
	python manage.py runserver

shell:
	python manage.py shell_plus

dbshell:
	python manage.py dbshell

# ============================================
# Database
# ============================================

migrate:
	python manage.py migrate

makemigrations:
	python manage.py makemigrations

resetdb:
	@echo "WARNING: This will delete all data!"
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ]
	python manage.py flush --no-input
	python manage.py migrate

# ============================================
# Static Files
# ============================================

static:
	python manage.py collectstatic --noinput

css:
	npm run build:css

css-watch:
	npm run dev:css

# ============================================
# Testing
# ============================================

test:
	python manage.py test

test-cov:
	coverage run manage.py test
	coverage report
	coverage html

lint:
	flake8 .
	isort --check-only .
	black --check .

format:
	isort .
	black .

# ============================================
# Celery
# ============================================

celery:
	celery -A core worker -l info

celery-beat:
	celery -A core beat -l info --scheduler django_celery_beat.schedulers:DatabaseScheduler

# ============================================
# Docker
# ============================================

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-build:
	docker-compose build

docker-restart:
	docker-compose restart

# ============================================
# Utilities
# ============================================

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type f -name "*.log" -delete

superuser:
	python manage.py createsuperuser

messages:
	python manage.py makemessages -l bn -l en

compile:
	python manage.py compilemessages

# ============================================
# Production
# ============================================

deploy:
	git pull origin main
	pip install -r requirements.txt
	npm install --legacy-peer-deps
	npm run build:css
	python manage.py migrate
	python manage.py collectstatic --noinput
	sudo systemctl restart gunicorn
	sudo systemctl restart celery
	sudo systemctl restart celery-beat
