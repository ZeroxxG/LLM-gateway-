"""
config/celery_app.py
====================
Celery application factory — stub for Phase 6.

Importing this module initialises Celery with Django settings so tasks can
use the Django ORM and access settings.CELERY_* values.
"""
import os

from celery import Celery

# Celery reads Django settings for broker URL, result backend, etc.
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.development")

app = Celery("llm_gateway")

# Namespace CELERY_ means all Celery config keys must start with CELERY_
# This reads CELERY_BROKER_URL, CELERY_RESULT_BACKEND, etc. from settings.
app.config_from_object("django.conf:settings", namespace="CELERY")

# Auto-discover tasks.py in every INSTALLED_APPS module
app.autodiscover_tasks()
