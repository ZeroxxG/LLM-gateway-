"""
config/settings/development.py
================================
Development-environment settings.
Inherits all base settings and relaxes constraints for local DX.
"""
from config.settings.base import *  # noqa: F401, F403
from config.settings.base import LOGGING

# Full debug mode: Django returns detailed HTML error pages on exceptions
DEBUG = True

# Accept any Host header locally — restrictive in production
ALLOWED_HOSTS = ["*"]

# Verbose logging in development
LOGGING["root"]["level"] = "DEBUG"           # type: ignore[index]
LOGGING["loggers"]["gateway"]["level"] = "DEBUG"  # type: ignore[index]
