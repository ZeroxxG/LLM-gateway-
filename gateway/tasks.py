"""
gateway/tasks.py
================
Asynchronous Background Celery Tasks for LLM Gateway:
  - Asynchronous Usage & Audit Logging (offloaded from ASGI event loop)
  - Daily Usage Metrics Aggregation Rollup
  - FinOps Tenant Budget Reconciliation & Drift Detection
  - Audit Log Lifecycle Retention Pruning
"""
import asyncio
from datetime import date, datetime, timedelta
from decimal import Decimal
import logging
from typing import Any, Dict, Optional
import uuid

from celery import shared_task
from django.db.models import Count, Q, Sum
from django.utils import timezone

from gateway.models import DailyUsageAggregate, RequestAuditLog, Tenant

logger = logging.getLogger(__name__)


@shared_task(
    name="gateway.tasks.log_request_usage_task",
    bind=True,
    max_retries=3,
    default_retry_delay=5,
    acks_late=True,
)
def log_request_usage_task(self, audit_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Asynchronously persists a RequestAuditLog record in PostgreSQL.
    Executed in the background by Celery workers to keep ASGI request-response latency ultra-low.
    """
    try:
        tenant_id = audit_data["tenant_id"]
        tenant = Tenant.objects.get(pk=tenant_id)

        cost_val = Decimal(str(audit_data.get("cost_usd", "0.00000000")))

        log = RequestAuditLog.objects.create(
            tenant=tenant,
            request_id=audit_data.get("request_id", str(uuid.uuid4())),
            model_id=audit_data.get("model_id", "unknown"),
            provider=audit_data.get("provider", "unknown"),
            prompt_tokens=audit_data.get("prompt_tokens", 0),
            completion_tokens=audit_data.get("completion_tokens", 0),
            total_tokens=audit_data.get("total_tokens", 0),
            cost_usd=cost_val,
            latency_ms=audit_data.get("latency_ms", 0.0),
            status_code=audit_data.get("status_code", 200),
            error_message=audit_data.get("error_message"),
            routing_strategy=audit_data.get("routing_strategy", "direct"),
            fallback_occurred=audit_data.get("fallback_occurred", False),
        )

        logger.debug(
            "Async audit log recorded: %s (tenant=%s, model=%s, tokens=%d, cost=$%s)",
            log.id,
            tenant.name,
            log.model_id,
            log.total_tokens,
            log.cost_usd,
        )

        return {
            "status": "success",
            "audit_log_id": str(log.id),
            "request_id": log.request_id,
            "tenant_id": str(tenant.id),
            "total_tokens": log.total_tokens,
            "cost_usd": str(log.cost_usd),
        }

    except Tenant.DoesNotExist:
        logger.error("Audit logging failed: Tenant with ID '%s' does not exist", audit_data.get("tenant_id"))
        return {"status": "error", "message": f"Tenant {audit_data.get('tenant_id')} not found"}
    except Exception as exc:
        logger.exception("Error in log_request_usage_task: %s. Retrying...", exc)
        raise self.retry(exc=exc)


@shared_task(name="gateway.tasks.aggregate_daily_usage_task", acks_late=True)
def aggregate_daily_usage_task(target_date_str: Optional[str] = None) -> Dict[str, Any]:
    """
    Rolls up RequestAuditLog records for a specific day into DailyUsageAggregate rows.
    Enables instant analytics queries without scanning millions of granular audit log rows.
    """
    if target_date_str:
        target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()
    else:
        target_date = timezone.now().date()

    start_dt = timezone.make_aware(datetime.combine(target_date, datetime.min.time()))
    end_dt = timezone.make_aware(datetime.combine(target_date, datetime.max.time()))

    # Find distinct tenants active on target date
    active_tenants = Tenant.objects.filter(
        audit_logs__created_at__range=(start_dt, end_dt)
    ).distinct()

    aggregated_count = 0

    for tenant in active_tenants:
        logs_qs = RequestAuditLog.objects.filter(
            tenant=tenant,
            created_at__range=(start_dt, end_dt),
        )

        metrics = logs_qs.aggregate(
            total_requests=Count("id"),
            successful_requests=Count("id", filter=Q(status_code=200)),
            failed_requests=Count("id", filter=~Q(status_code=200)),
            total_prompt_tokens=Sum("prompt_tokens"),
            total_completion_tokens=Sum("completion_tokens"),
            total_tokens=Sum("total_tokens"),
            total_cost_usd=Sum("cost_usd"),
        )

        DailyUsageAggregate.objects.update_or_create(
            tenant=tenant,
            date=target_date,
            defaults={
                "total_requests": metrics["total_requests"] or 0,
                "successful_requests": metrics["successful_requests"] or 0,
                "failed_requests": metrics["failed_requests"] or 0,
                "total_prompt_tokens": metrics["total_prompt_tokens"] or 0,
                "total_completion_tokens": metrics["total_completion_tokens"] or 0,
                "total_tokens": metrics["total_tokens"] or 0,
                "total_cost_usd": (metrics["total_cost_usd"] or Decimal("0.000000")).quantize(Decimal("0.000001")),
            },
        )
        aggregated_count += 1

    logger.info("Daily usage aggregated for %s: %d tenants processed", target_date, aggregated_count)
    return {
        "status": "success",
        "date": str(target_date),
        "tenants_aggregated": aggregated_count,
    }


@shared_task(name="gateway.tasks.reconcile_tenant_budgets_task", acks_late=True)
def reconcile_tenant_budgets_task() -> Dict[str, Any]:
    """
    FinOps audit job: Compares aggregate audit log spending with Tenant.current_spend_usd.
    Identifies any discrepancies or drift between logged requests and tenant balance.
    """
    active_tenants = Tenant.objects.filter(is_active=True)
    reconciled_results = []
    discrepancies_found = 0

    for tenant in active_tenants:
        total_logged = RequestAuditLog.objects.filter(tenant=tenant).aggregate(
            total=Sum("cost_usd")
        )["total"] or Decimal("0.00000000")

        logged_dec = Decimal(str(total_logged)).quantize(Decimal("0.000001"))
        spend_dec = tenant.current_spend_usd.quantize(Decimal("0.000001"))
        drift = abs(logged_dec - spend_dec)

        has_drift = drift > Decimal("0.001000")
        if has_drift:
            discrepancies_found += 1
            logger.warning(
                "FinOps budget drift detected for tenant '%s': spend=$%s vs logged=$%s (drift=$%s)",
                tenant.name,
                spend_dec,
                logged_dec,
                drift,
            )

        reconciled_results.append({
            "tenant_id": str(tenant.id),
            "tenant_name": tenant.name,
            "current_spend_usd": str(spend_dec),
            "logged_spend_usd": str(logged_dec),
            "drift_usd": str(drift),
            "reconciled": not has_drift,
        })

    return {
        "status": "success",
        "tenants_checked": len(reconciled_results),
        "discrepancies_found": discrepancies_found,
        "details": reconciled_results,
    }


@shared_task(name="gateway.tasks.cleanup_old_audit_logs_task", acks_late=True)
def cleanup_old_audit_logs_task(retention_days: int = 30) -> Dict[str, Any]:
    """
    Maintenance task to prune RequestAuditLog rows older than the specified retention period.
    """
    cutoff = timezone.now() - timedelta(days=retention_days)
    deleted_count, _ = RequestAuditLog.objects.filter(created_at__lt=cutoff).delete()
    logger.info("Cleaned up %d audit logs older than %d days (cutoff: %s)", deleted_count, retention_days, cutoff)
    return {
        "status": "success",
        "deleted_count": deleted_count,
        "cutoff": str(cutoff),
    }


async def dispatch_audit_log_async(audit_data: Dict[str, Any]) -> None:
    """
    Pure async helper to dispatch audit log recording to Celery without blocking ASGI event loop.
    Uses asyncio.to_thread to run kombu Redis LPUSH safely in a worker thread.
    """
    try:
        await asyncio.to_thread(log_request_usage_task.delay, audit_data)
    except Exception as exc:
        logger.exception("Failed to dispatch Celery audit log task: %s", exc)
