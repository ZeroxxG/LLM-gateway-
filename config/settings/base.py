"""
config/settings/base.py
=======================
Base Django settings shared across all environments.

Architecture:
  - Pure API server: admin, sessions, staticfiles intentionally excluded.
  - Django 5.x + psycopg3 enables async ORM (acreate, afilter, aget, etc.).
  - All secrets/URIs injected via environment variables — zero hardcoded values.
  - django-environ parses DATABASE_URL into Django's DATABASES dict format.
"""
import os
from pathlib import Path

import environ

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
# BASE_DIR → repository root:
#   config/settings/base.py  →  config/settings/  →  config/  →  repo root
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# django-environ initialisation
# Declares type casts and safe defaults for each env var.
# ---------------------------------------------------------------------------
env = environ.Env(
    DEBUG=(bool, False),
    ALLOWED_HOSTS=(list, ["*"]),
    REDIS_URL=(str, "redis://localhost:6379/0"),
    DEFAULT_ROUTING_STRATEGY=(str, "cost_optimized"),
    MAX_RETRIES=(int, 3),
    REQUEST_TIMEOUT_SECONDS=(int, 30),
)

# Silently no-ops when .env is absent (prod uses real OS env vars)
environ.Env.read_env(os.path.join(BASE_DIR, ".env"))

# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------
SECRET_KEY: str = env("SECRET_KEY")
DEBUG: bool = env("DEBUG")
ALLOWED_HOSTS: list[str] = env("ALLOWED_HOSTS")

# ---------------------------------------------------------------------------
# Application definition — minimal, API-only
# ---------------------------------------------------------------------------
INSTALLED_APPS: list[str] = [
    # Django internals (contenttypes + auth needed for migrations / ORM)
    "django.contrib.contenttypes",
    "django.contrib.auth",
    # Our gateway app
    "gateway",
]

MIDDLEWARE: list[str] = [
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
]

ROOT_URLCONF = "config.urls"

# ASGI_APPLICATION tells Django Channels / Uvicorn which callable to serve.
ASGI_APPLICATION = "config.asgi.application"

# ---------------------------------------------------------------------------
# Database — Django 5.x + psycopg3 (full async ORM support)
# ---------------------------------------------------------------------------
# django-environ parses DATABASE_URL → produces a dict compatible with DATABASES.
# We force ENGINE to django.db.backends.postgresql so Django picks psycopg3.
# psycopg3's async driver is then used by async ORM calls (afilter, aget, etc.)
_db_config = env.db(
    "DATABASE_URL",
    default="postgres://gateway_user:gateway_pass@localhost:5432/llm_gateway",
)
_db_config["ENGINE"] = "django.db.backends.postgresql"  # psycopg3 auto-selected
_db_config["OPTIONS"] = {
    "connect_timeout": 10,
    "sslmode": env("DB_SSLMODE", default="disable"),
}
_db_config["CONN_MAX_AGE"] = 60           # Keep connections alive for 60 s
_db_config["CONN_HEALTH_CHECKS"] = True   # Re-validate connection before reuse

DATABASES: dict = {"default": _db_config}

# ---------------------------------------------------------------------------
# Redis — raw URL stored for direct use by the async redis client
# ---------------------------------------------------------------------------
REDIS_URL: str = env("REDIS_URL")

# ---------------------------------------------------------------------------
# Django Cache Framework — backed by Redis (used for rate limiting in Phase 5)
# ---------------------------------------------------------------------------
CACHES: dict = {
    "default": {
        "BACKEND": "django.core.cache.backends.redis.RedisCache",
        "LOCATION": REDIS_URL,
    }
}

# ---------------------------------------------------------------------------
# Celery — broker + result backend both on Redis
# ---------------------------------------------------------------------------
CELERY_BROKER_URL: str = REDIS_URL
CELERY_RESULT_BACKEND: str = REDIS_URL
CELERY_ACCEPT_CONTENT: list[str] = ["json"]
CELERY_TASK_SERIALIZER: str = "json"
CELERY_RESULT_SERIALIZER: str = "json"
CELERY_TIMEZONE: str = "UTC"
CELERY_TASK_TRACK_STARTED: bool = True
CELERY_RESULT_EXTENDED: bool = True
CELERY_TASK_ALWAYS_EAGER: bool = env.bool("CELERY_TASK_ALWAYS_EAGER", default=False)
CELERY_TASK_EAGER_PROPAGATES: bool = True
CELERY_BROKER_CONNECTION_RETRY_ON_STARTUP: bool = True

# ---------------------------------------------------------------------------
# Internationalisation — disabled (pure API, no i18n)
# ---------------------------------------------------------------------------
LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = False
USE_TZ = True

# ---------------------------------------------------------------------------
# Default primary key field type
# ---------------------------------------------------------------------------
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# ---------------------------------------------------------------------------
# LLM Gateway application settings
# ---------------------------------------------------------------------------
GATEWAY_DEFAULT_ROUTING_STRATEGY: str = env("DEFAULT_ROUTING_STRATEGY")
GATEWAY_MAX_RETRIES: int = env("MAX_RETRIES")
GATEWAY_REQUEST_TIMEOUT_SECONDS: int = env("REQUEST_TIMEOUT_SECONDS")

# LLM provider API keys (empty strings OK for Phase 1; validated in Phase 3)
OPENAI_API_KEY: str = env("OPENAI_API_KEY", default="")
GROQ_API_KEY: str = env("GROQ_API_KEY", default="")
GEMINI_API_KEY: str = env("GEMINI_API_KEY", default="")

# Langfuse telemetry (Phase 7)
LANGFUSE_PUBLIC_KEY: str = env("LANGFUSE_PUBLIC_KEY", default="")
LANGFUSE_SECRET_KEY: str = env("LANGFUSE_SECRET_KEY", default="")
LANGFUSE_HOST: str = env("LANGFUSE_HOST", default="https://cloud.langfuse.com")

# ---------------------------------------------------------------------------
# Structured logging — verbose format for development; JSON in production
# ---------------------------------------------------------------------------
LOGGING: dict = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
        },
        "verbose": {
            "format": "[{levelname}] {asctime} {name} — {message}",
            "style": "{",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "verbose",
        },
    },
    "root": {
        "handlers": ["console"],
        "level": "INFO",
    },
    "loggers": {
        "django":   {"handlers": ["console"], "level": "WARNING", "propagate": False},
        "gateway":  {"handlers": ["console"], "level": "DEBUG",   "propagate": False},
        "uvicorn":  {"handlers": ["console"], "level": "INFO",    "propagate": False},
    },
}
