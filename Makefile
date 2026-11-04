# =============================================================================
# LLM Gateway — Makefile
# =============================================================================
.PHONY: help install env up down logs shell migrate test lint

help:
	@echo ""
	@echo "LLM Gateway — Available Commands"
	@echo "---------------------------------"
	@echo "  make install   Install Python dependencies into venv"
	@echo "  make env       Copy .env.example → .env (first-time setup)"
	@echo "  make up        Start all Docker services (detached)"
	@echo "  make down      Stop all Docker services"
	@echo "  make logs      Tail API logs"
	@echo "  make shell     Open Django shell inside API container"
	@echo "  make migrate   Run Django migrations"
	@echo "  make test      Run test suite"
	@echo "  make lint      Run ruff linter"
	@echo ""

install:
	python -m venv .venv
	.venv/Scripts/pip install -r requirements.txt

env:
	copy .env.example .env
	@echo ".env created. Edit it and add your API keys."

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f api

shell:
	docker compose exec api python manage.py shell

migrate:
	docker compose exec api python manage.py migrate

test:
	docker compose exec api python -m pytest

lint:
	.venv/Scripts/ruff check .
