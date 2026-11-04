"""
gateway/management/commands/seed_tenants_and_pricing.py
=======================================================
Django management command to bootstrap test tenant, API key, and model pricing.

Usage:
  python manage.py seed_tenants_and_pricing
  python manage.py seed_tenants_and_pricing --reset-key
"""
from decimal import Decimal

from django.core.management.base import BaseCommand
from django.utils import timezone

from gateway.models import APIKey, ModelPricing, ModelTier, Tenant, TenantTier
from gateway.security import generate_api_key


class Command(BaseCommand):
    help = "Seeds initial test Tenant, generates an active API Key, and populates the Model Pricing catalog."

    def add_arguments(self, parser):
        parser.add_argument(
            "--reset-key",
            action="store_true",
            help="Generate a new active API key even if one already exists for the test tenant.",
        )

    def handle(self, *args, **options):
        self.stdout.write(self.style.MIGRATE_HEADING("=== [Phase 2] Seeding Tenants, API Keys & Model Pricing ==="))

        # ---------------------------------------------------------------------
        # 1. Seed or Update Test Tenant
        # ---------------------------------------------------------------------
        tenant_name = "Acme Global (Test Org)"
        tenant, created = Tenant.objects.update_or_create(
            name=tenant_name,
            defaults={
                "tier": TenantTier.STANDARD,
                "monthly_budget_usd": Decimal("50.0000"),
                "current_spend_usd": Decimal("0.0000"),
                "rpm_limit": 120,
                "tpm_limit": 100000,
                "is_active": True,
            },
        )

        status_label = "Created new" if created else "Updated existing"
        self.stdout.write(
            self.style.SUCCESS(
                f"[Tenant] {status_label} tenant: '{tenant.name}' (ID: {tenant.id}) with budget ${tenant.monthly_budget_usd}"
            )
        )

        # ---------------------------------------------------------------------
        # 2. Seed API Key
        # ---------------------------------------------------------------------
        existing_keys = APIKey.objects.filter(tenant=tenant, is_active=True)
        should_generate_key = options["reset_key"] or not existing_keys.exists()

        if should_generate_key:
            raw_key, prefix, hashed_key = generate_api_key()
            api_key_obj = APIKey.objects.create(
                tenant=tenant,
                key_prefix=prefix,
                hashed_key=hashed_key,
                name="Phase 2 Dev Test Key",
                is_active=True,
            )

            self.stdout.write("\n" + "=" * 70)
            self.stdout.write(self.style.WARNING("[KEY] NEW API KEY GENERATED (SAVE THIS IN YOUR .env OR TEST RUNNER):"))
            self.stdout.write(self.style.NOTICE(f"   Raw API Key:  {raw_key}"))
            self.stdout.write(f"   Key Prefix:   {prefix}")
            self.stdout.write(f"   Key ID:       {api_key_obj.id}")
            self.stdout.write(f"   Hashed (DB):  {hashed_key[:16]}...{hashed_key[-8:]}")
            self.stdout.write(
                self.style.WARNING(
                    "[NOTE] The raw key is never stored in the database and CANNOT be retrieved again."
                )
            )
            self.stdout.write("=" * 70 + "\n")
        else:
            first_key = existing_keys.first()
            self.stdout.write(
                self.style.WARNING(
                    f"[API Key] Active key already exists (Prefix: {first_key.key_prefix}..., Name: '{first_key.name}').\n"
                    f"          Pass --reset-key to generate a new plaintext key token."
                )
            )

        # ---------------------------------------------------------------------
        # 3. Seed Model Pricing Catalog
        # ---------------------------------------------------------------------
        catalog_items = [
            {
                "model_id": "groq/llama-3.1-8b",
                "provider": "groq",
                "input_cost_per_million": Decimal("0.0500"),
                "output_cost_per_million": Decimal("0.0800"),
                "tier": ModelTier.SMALL,
                "is_active": True,
            },
            {
                "model_id": "openai/gpt-4o-mini",
                "provider": "openai",
                "input_cost_per_million": Decimal("0.1500"),
                "output_cost_per_million": Decimal("0.6000"),
                "tier": ModelTier.SMALL,
                "is_active": True,
            },
            {
                "model_id": "anthropic/claude-3-5-sonnet",
                "provider": "anthropic",
                "input_cost_per_million": Decimal("3.0000"),
                "output_cost_per_million": Decimal("15.0000"),
                "tier": ModelTier.PREMIUM,
                "is_active": True,
            },
            {
                "model_id": "google/gemini-1.5-flash",
                "provider": "gemini",
                "input_cost_per_million": Decimal("0.0750"),
                "output_cost_per_million": Decimal("0.3000"),
                "tier": ModelTier.SMALL,
                "is_active": True,
            },
        ]

        self.stdout.write("\n[Model Pricing Catalog]")
        for item in catalog_items:
            obj, is_new = ModelPricing.objects.update_or_create(
                model_id=item["model_id"],
                defaults=item,
            )
            action = "Created" if is_new else "Synchronized"
            self.stdout.write(
                f"  - {action} '{obj.model_id}': ${obj.input_cost_per_million}/M input | ${obj.output_cost_per_million}/M output ({obj.tier})"
            )

        self.stdout.write(self.style.SUCCESS("\n[SUCCESS] [Phase 2] Seeding completed successfully."))
