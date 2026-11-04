"""
api/models.py
=============
Re-exports models from gateway.models for namespace compatibility.
"""
from gateway.models import APIKey, ModelPricing, ModelTier, Tenant, TenantTier

__all__ = ["Tenant", "APIKey", "ModelPricing", "TenantTier", "ModelTier"]
