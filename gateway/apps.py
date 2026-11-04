# gateway/apps.py
from django.apps import AppConfig


class GatewayConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "gateway"
    verbose_name = "LLM Gateway"
