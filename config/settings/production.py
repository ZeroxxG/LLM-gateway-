"""
config/settings/production.py
================================
Production-environment settings.
Tightens security, disables debug, and switches to JSON logging.
"""
from config.settings.base import *  # noqa: F401, F403
from config.settings.base import LOGGING

DEBUG = False

# In production set this to your actual domain(s) via env var ALLOWED_HOSTS
# e.g. ALLOWED_HOSTS=gateway.example.com,api.example.com

# Switch log formatter to JSON for ingestion by log aggregators (Datadog, GCP, etc.)
LOGGING["handlers"]["console"]["formatter"] = "json"  # type: ignore[index]
LOGGING["root"]["level"] = "INFO"                      # type: ignore[index]
