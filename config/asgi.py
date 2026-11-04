"""
config/asgi.py
==============
ASGI-3 entrypoint for the LLM Gateway.

Django runs ASGI-first — there is no wsgi.py in this project.
Uvicorn boots this module directly:

    uvicorn config.asgi:application --host 0.0.0.0 --port 8000

Startup sequence:
  1. os.environ sets DJANGO_SETTINGS_MODULE before any Django import.
  2. get_asgi_application() calls django.setup() internally, which:
       a. Imports and validates settings
       b. Populates the app registry (INSTALLED_APPS)
       c. Registers signal handlers
  3. Returns an ASGI-3 callable: async def app(scope, receive, send)
  4. Uvicorn calls this callable for every incoming HTTP connection.

Why ASGI over WSGI?
  WSGI is synchronous — each request occupies a thread until completion.
  ASGI is async — a single thread multiplexes thousands of concurrent I/O-
  bound requests using the event loop, which is critical for our use case
  (long-polling LLM provider calls without exhausting thread pools).
"""
import os

from django.core.asgi import get_asgi_application

# Must be set before any Django import so settings are available module-wide
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.development")

# The ASGI-3 callable that Uvicorn binds to.
# Django Ninja's async views are dispatched through this callable.
application = get_asgi_application()
