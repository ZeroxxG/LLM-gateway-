"""
tests/test_helpers.py
=====================
Helper functions to seed test database for isolated pytest test runs.
"""
from decimal import Decimal
from gateway.models import APIKey, ModelPricing, ModelTier, Tenant, TenantTier
from gateway.security import hash_api_key

KNOWN_TEST_KEY = "gw_live_e6ed9d77b3841b97dadd97bb030f4f5a"


async def ensure_seed_data() -> Tenant:
    """Ensure baseline tenant, API key, and model pricing exist in the test DB."""
    tenant, _ = await Tenant.objects.aupdate_or_create(
        name="Acme Global (Test Org)",
        defaults={
            "tier": TenantTier.STANDARD,
            "monthly_budget_usd": Decimal("50.000000"),
            "current_spend_usd": Decimal("0.000000"),
            "rpm_limit": 120,
            "tpm_limit": 100000,
            "is_active": True,
        },
    )

    hashed = hash_api_key(KNOWN_TEST_KEY)
    await APIKey.objects.aupdate_or_create(
        tenant=tenant,
        hashed_key=hashed,
        defaults={
            "key_prefix": KNOWN_TEST_KEY[:8],
            "name": "Phase 2 Dev Test Key",
            "is_active": True,
        },
    )

    pricings = [
        {"model_id": "mock/test-model", "provider": "mock", "input_cost": "0.1000", "output_cost": "0.2000", "tier": ModelTier.SMALL},
        {"model_id": "groq/llama-3.1-8b", "provider": "groq", "input_cost": "0.0500", "output_cost": "0.0800", "tier": ModelTier.SMALL},
        {"model_id": "openai/gpt-4o-mini", "provider": "openai", "input_cost": "0.1500", "output_cost": "0.6000", "tier": ModelTier.MEDIUM},
        {"model_id": "anthropic/claude-3-5-sonnet", "provider": "anthropic", "input_cost": "3.0000", "output_cost": "15.0000", "tier": ModelTier.PREMIUM},
        {"model_id": "google/gemini-1.5-flash", "provider": "gemini", "input_cost": "0.0750", "output_cost": "0.3000", "tier": ModelTier.SMALL},
    ]
    for p in pricings:
        await ModelPricing.objects.aupdate_or_create(
            model_id=p["model_id"],
            defaults={
                "provider": p["provider"],
                "input_cost_per_million": Decimal(p["input_cost"]),
                "output_cost_per_million": Decimal(p["output_cost"]),
                "tier": p["tier"],
                "is_active": True,
            },
        )
    return tenant
