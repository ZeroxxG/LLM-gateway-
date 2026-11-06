# Intelligent Multi-Model LLM Gateway

Enterprise-grade, pure ASGI multi-tenant LLM Gateway built with Django 5.x, Django Ninja, PostgreSQL 16, Redis 7, Celery, and Prometheus.

---

## Architecture Overview

```
                      ┌───────────────────────────────────────────────┐
                      │              Client Applications              │
                      └───────────────────────┬───────────────────────┘
                                              │ Bearer / X-API-Key Auth
                                              ▼
                      ┌───────────────────────────────────────────────┐
                      │        ASGI Django Ninja Gateway Core         │
                      ├───────────────────────────────────────────────┤
                      │  - SHA-256 Hashed API Key Verification        │
                      │  - Real-time Redis FinOps Pre-flight (402)    │
                      │  - Sliding-Window Rate Limiter (RPM / TPM)    │
                      │  - Deterministic Prompt Response Cache        │
                      │  - Sub-2ms Heuristic Dynamic Decision Engine  │
                      │  - Distributed Redis Circuit Breakers         │
                      └───────┬───────────────────────────────┬───────┘
                              │                               │
                Stream / Sync │                               │ Metrics & Logs
                              ▼                               ▼
       ┌───────────────────────────────┐        ┌─────────────────────────────┐
       │   Multi-Provider Adapters     │        │ Prometheus Scrape (/metrics)│
       ├───────────────────────────────┤        │ Celery Async Audit Logs     │
       │ Groq (LPU)                    │        │ Sliding-window Latency Quant│
       │ OpenAI (GPT-4o / Mini)        │        │ Langfuse Telemetry Spans    │
       │ Gemini (Flash / Pro)          │        └─────────────────────────────┘
       │ Mock (Deterministic Engine)   │
       └───────────────────────────────┘
```

---

## Core Capabilities by Phase

1. **Phase 1: Pure ASGI Scaffold & Infrastructure**: High-concurrency async foundation with concurrent PostgreSQL and Redis health probes.
2. **Phase 2: Multi-Tenant Isolation & SHA-256 Auth**: Cryptographically secure tenant isolation with custom `APIKeyAuth` and active model token pricing catalog.
3. **Phase 3: Unified Provider Adapters & FinOps Token Accounting**: Standardized schemas for Groq, OpenAI, Gemini, and Mock adapters with automatic per-request dollar cost calculation.
4. **Phase 4: Dynamic Routing & Distributed Circuit Breakers**: Distributed state machines (`CLOSED`, `OPEN`, `HALF_OPEN`) preventing cascading provider outages with multi-candidate fallback orchestration.
5. **Phase 5: Sliding-Window Rate Limiting (RPM/TPM)**: Atomic Redis Lua sliding-window rate limiter with RFC-compliant headers and soft budget threshold warnings.
6. **Phase 6: Asynchronous Celery Workers & Analytics**: Non-blocking audit logging, daily usage rollups, periodic budget reconciliations, and retention cleanups.
7. **Phase 7: Real-Time Atomic Spend Tracking**: Sub-millisecond pre-flight Redis checks enforcing hard-stop HTTP 402 lockouts when tenant spending limits are exceeded.
8. **Phase 8: Sub-2ms Heuristic Decision Engine**: Regex task classification, token estimation, SLA constraint filters, and multi-factor capability scoring.
9. **Phase 9: Server-Sent Events (SSE) Streaming**: Real-time token streaming returning `data: {...}` chunks and `data: [DONE]` with pre-first-chunk fallback resilience.
10. **Phase 10: Prompt Response Caching & Acceleration**: Deterministic SHA-256 response caching delivering sub-10ms cache hits with `X-Cache: HIT / MISS / BYPASS` headers.
11. **Phase 11: Observability, Prometheus Metrics & Distributed Tracing**: Prometheus exposition endpoints (`/metrics`), sliding-window empirical latency percentiles ($p50, p90, p95, p99$), and distributed `X-Trace-Id` propagation.
12. **Phase 12: Benchmark Harness & Baseline Evaluation**: Curated 40-item comparative test suite proving a **78.79% spend cut** (4.72x cost efficiency) while maintaining **92.5% accuracy** against the All-Premium baseline.

---

## Benchmark Evaluation Results

The 40-item benchmark compares routing all traffic to the Premium model (`anthropic/claude-3-5-sonnet`) versus the Gateway's Intelligent Heuristic Router:

| Metric | All-Premium Baseline | Routed Gateway | Improvement / Delta |
| :--- | :--- | :--- | :--- |
| **Total Workload** | 40 items | 40 items | Equal test suite |
| **Total Spend** | **\$0.094356** | **\$0.020010** | **-78.79% spend cut** |
| **Cost Efficiency** | 1.0x (baseline) | **4.72x cheaper** | **\$4.72 value per \$1 spent** |
| **Accuracy Rate** | 100.0% | **92.5%** | **-7.5% retention** |
| **Average Latency** | 1,400.0 ms | **292.0 ms** | **-79.14% faster** |

---

## Quickstart

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/ZeroxxG/LLM-gateway-.git
cd LLM-gateway-

python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

### 2. Launch Infrastructure via Docker
```bash
docker compose up -d db redis
```

### 3. Run Migrations & Start Server
```bash
python manage.py migrate
uvicorn config.asgi:application --reload --port 8000
```

---

## API Reference

### Completions & Routing
- `POST /api/v1/chat/completions`: Unified chat completion endpoint supporting standard and streaming (`stream: true`) requests with rate limits, caching, and fallback routing.
- `POST /api/v1/routing/evaluate`: Sub-2ms heuristic routing diagnostic test endpoint.
- `GET /api/v1/routing/circuits`: Live inspection of provider circuit breaker states.
- `POST /api/v1/routing/circuits/{provider}/reset`: Reset provider circuit breaker back to CLOSED.

### Observability & Telemetry
- `GET /metrics`: Public Prometheus text format scrape endpoint.
- `GET /api/v1/observability/summary`: Authenticated summary of request volume, token usage, and empirical $p50, p90, p95, p99$ latency percentiles.

### Benchmark Harness
- `POST /api/v1/benchmarks/run`: Run comparative benchmark suite comparing All-Premium vs. Routed Gateway.

### Cache & Acceleration
- `GET /api/v1/cache/stats`: Active cache key counts, hits, misses, and hit ratio.
- `POST /api/v1/cache/clear`: Invalidate cached prompt responses for tenant.
