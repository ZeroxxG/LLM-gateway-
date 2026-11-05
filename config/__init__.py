# config/__init__.py
# Makes config/ a Python package and ensures Celery app is loaded on Django startup
from .celery_app import app as celery_app

__all__ = ("celery_app",)
