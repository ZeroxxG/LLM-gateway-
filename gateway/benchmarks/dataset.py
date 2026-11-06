"""
gateway/benchmarks/dataset.py
=============================
Curated 40-item benchmark dataset for LLM Gateway evaluation.

Spans 5 key task categories across low, medium, and high complexity:
  - Simple factual queries & chat (10 items) -> Low complexity (ideal for Small tier)
  - Extraction & structured JSON (10 items) -> Medium complexity (ideal for Small/Medium tier)
  - Summarization & creative rewrite (10 items) -> Medium complexity (ideal for Medium tier)
  - Advanced coding & reasoning (10 items) -> High complexity (requires Premium tier)
"""
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class BenchmarkItem:
    """A single benchmark test case."""

    id: str
    category: str  # chat, extraction, summarization, coding, reasoning
    complexity: str  # low, medium, high
    prompt: str
    expected_keywords: List[str]
    required_tier: str  # small, medium, premium


BENCHMARK_DATASET: List[BenchmarkItem] = [
    # -------------------------------------------------------------------------
    # 1. Simple Queries & Chat (10 Items) - Low Complexity
    # -------------------------------------------------------------------------
    BenchmarkItem(
        id="bench-01",
        category="chat",
        complexity="low",
        prompt="What is the capital city of France?",
        expected_keywords=["paris"],
        required_tier="small",
    ),
    BenchmarkItem(
        id="bench-02",
        category="chat",
        complexity="low",
        prompt="How many continents are there on planet Earth?",
        expected_keywords=["7", "seven"],
        required_tier="small",
    ),
    BenchmarkItem(
        id="bench-03",
        category="chat",
        complexity="low",
        prompt="Translate the English greeting 'Good morning' into Spanish.",
        expected_keywords=["buenos", "dias"],
        required_tier="small",
    ),
    BenchmarkItem(
        id="bench-04",
        category="chat",
        complexity="low",
        prompt="What is the chemical formula for pure water?",
        expected_keywords=["h2o"],
        required_tier="small",
    ),
    BenchmarkItem(
        id="bench-05",
        category="chat",
        complexity="low",
        prompt="Name the three primary colors in traditional subtractive color theory.",
        expected_keywords=["red", "blue", "yellow"],
        required_tier="small",
    ),
    BenchmarkItem(
        id="bench-06",
        category="chat",
        complexity="low",
        prompt="What is the boiling point of pure water at sea level in Celsius?",
        expected_keywords=["100"],
        required_tier="small",
    ),
    BenchmarkItem(
        id="bench-07",
        category="chat",
        complexity="low",
        prompt="Who painted the famous Mona Lisa portrait?",
        expected_keywords=["leonardo", "da vinci"],
        required_tier="small",
    ),
    BenchmarkItem(
        id="bench-08",
        category="chat",
        complexity="low",
        prompt="What is the largest planet in our solar system?",
        expected_keywords=["jupiter"],
        required_tier="small",
    ),
    BenchmarkItem(
        id="bench-09",
        category="chat",
        complexity="low",
        prompt="What is the direct antonym of the word 'expand'?",
        expected_keywords=["contract", "shrink"],
        required_tier="small",
    ),
    BenchmarkItem(
        id="bench-10",
        category="chat",
        complexity="low",
        prompt="How many sides does a regular two-dimensional hexagon have?",
        expected_keywords=["6", "six"],
        required_tier="small",
    ),

    # -------------------------------------------------------------------------
    # 2. Data Extraction & Classification (10 Items) - Medium Complexity
    # -------------------------------------------------------------------------
    BenchmarkItem(
        id="bench-11",
        category="extraction",
        complexity="medium",
        prompt="Extract the person's name and contact email: 'Please reach out to Alice Smith at alice.smith@corp.com for assistance.'",
        expected_keywords=["alice", "smith", "alice.smith@corp.com"],
        required_tier="small",
    ),
    BenchmarkItem(
        id="bench-12",
        category="extraction",
        complexity="medium",
        prompt="Classify customer sentiment as positive, negative, or neutral: 'The latency was unbearable and API calls repeatedly timed out.'",
        expected_keywords=["negative"],
        required_tier="small",
    ),
    BenchmarkItem(
        id="bench-13",
        category="extraction",
        complexity="medium",
        prompt="Extract order number and total price: 'Order #94827 confirmed with payment of $149.99 via credit card.'",
        expected_keywords=["94827", "149.99"],
        required_tier="small",
    ),
    BenchmarkItem(
        id="bench-14",
        category="extraction",
        complexity="medium",
        prompt="Extract the departure city and destination city: 'United flight 429 from Seattle to Boston departs tomorrow at 9 AM.'",
        expected_keywords=["seattle", "boston"],
        required_tier="small",
    ),
    BenchmarkItem(
        id="bench-15",
        category="extraction",
        complexity="medium",
        prompt="Classify the support ticket intent into billing, technical, or sales: 'How much does it cost to upgrade our team to the enterprise plan?'",
        expected_keywords=["sales", "billing"],
        required_tier="small",
    ),
    BenchmarkItem(
        id="bench-16",
        category="extraction",
        complexity="medium",
        prompt="Format the date into ISO standard YYYY-MM-DD: 'The product launch conference takes place on October 24th, 2026.'",
        expected_keywords=["2026-10-24"],
        required_tier="medium",
    ),
    BenchmarkItem(
        id="bench-17",
        category="extraction",
        complexity="medium",
        prompt="Extract technical skills from the resume snippet: 'The software engineer is proficient in Python, Docker, and PostgreSQL.'",
        expected_keywords=["python", "docker", "postgresql"],
        required_tier="small",
    ),
    BenchmarkItem(
        id="bench-18",
        category="extraction",
        complexity="medium",
        prompt="Classify issue severity as P0, P1, or P2: 'The main database connection pool has crashed and all customer logins are down.'",
        expected_keywords=["p0", "critical", "high"],
        required_tier="small",
    ),
    BenchmarkItem(
        id="bench-19",
        category="extraction",
        complexity="medium",
        prompt="Extract the promotional discount code: 'Apply promo code SAVE50 at checkout to receive fifty percent off your purchase.'",
        expected_keywords=["save50"],
        required_tier="small",
    ),
    BenchmarkItem(
        id="bench-20",
        category="extraction",
        complexity="medium",
        prompt="Identify the language of this sentence: 'Guten Tag, wir freuen uns über Ihren Besuch auf unserer Webseite.'",
        expected_keywords=["german", "deutsch"],
        required_tier="small",
    ),

    # -------------------------------------------------------------------------
    # 3. Summarization & Text Transformation (10 Items) - Medium Complexity
    # -------------------------------------------------------------------------
    BenchmarkItem(
        id="bench-21",
        category="summarization",
        complexity="medium",
        prompt="Summarize this Q3 report into bullet points: 'Q3 revenue climbed 25% year-over-year driven by multi-model gateway adoption. Operating expenses fell 8% thanks to response caching in Redis. We signed 40 enterprise customers.'",
        expected_keywords=["revenue", "25%", "gateway", "caching", "customers"],
        required_tier="medium",
    ),
    BenchmarkItem(
        id="bench-22",
        category="summarization",
        complexity="medium",
        prompt="Provide a one-sentence executive summary: 'Cloud infrastructure bills spiked due to unrouted GPT-4o usage. After implementing intelligent heuristic routing with circuit breakers, total LLM spend dropped by 72% without hurting accuracy.'",
        expected_keywords=["spend", "routing", "costs"],
        required_tier="medium",
    ),
    BenchmarkItem(
        id="bench-23",
        category="summarization",
        complexity="medium",
        prompt="Extract all action items by owner: 'Bob will provision the Redis cluster on Monday. Carol will update the OpenAPI documentation on Tuesday. Dave will configure Grafana dashboards.'",
        expected_keywords=["bob", "redis", "carol", "dave", "grafana"],
        required_tier="medium",
    ),
    BenchmarkItem(
        id="bench-24",
        category="summarization",
        complexity="medium",
        prompt="Write a concise customer alert announcing scheduled gateway maintenance on Sunday at 02:00 UTC with expected downtime of 10 minutes.",
        expected_keywords=["maintenance", "sunday", "utc", "downtime"],
        required_tier="medium",
    ),
    BenchmarkItem(
        id="bench-25",
        category="summarization",
        complexity="medium",
        prompt="Summarize GDPR data retention and deletion obligations for multi-tenant database architectures in three concise bullet points.",
        expected_keywords=["retention", "deletion", "tenant"],
        required_tier="medium",
    ),
    BenchmarkItem(
        id="bench-26",
        category="summarization",
        complexity="medium",
        prompt="Explain why Server-Sent Events (SSE) are simpler and more suitable than WebSockets for uni-directional LLM token streaming.",
        expected_keywords=["http", "unidirectional", "streaming", "reconnect"],
        required_tier="medium",
    ),
    BenchmarkItem(
        id="bench-27",
        category="summarization",
        complexity="medium",
        prompt="Condense this postmortem: 'Service was degraded for 15 minutes due to upstream provider rate limits. The Redis circuit breaker tripped to OPEN, seamlessly failing over to backup provider.'",
        expected_keywords=["degraded", "rate limit", "circuit breaker", "failover"],
        required_tier="medium",
    ),
    BenchmarkItem(
        id="bench-28",
        category="summarization",
        complexity="medium",
        prompt="Rewrite this cryptic error code 'ERR_TPM_EXCEEDED_LOCK' into a friendly message for an API consumer.",
        expected_keywords=["token", "rate limit", "exceeded", "minute"],
        required_tier="medium",
    ),
    BenchmarkItem(
        id="bench-29",
        category="summarization",
        complexity="medium",
        prompt="Write a clear Git commit message describing the implementation of Prometheus metrics and p50/p95 latency quantiles.",
        expected_keywords=["prometheus", "metrics", "latency"],
        required_tier="small",
    ),
    BenchmarkItem(
        id="bench-30",
        category="summarization",
        complexity="medium",
        prompt="Explain Redis sliding-window rate limiting using sorted sets in two plain sentences.",
        expected_keywords=["sliding", "window", "timestamp", "redis"],
        required_tier="medium",
    ),

    # -------------------------------------------------------------------------
    # 4. Complex Reasoning & Code Architecture (10 Items) - High Complexity
    # -------------------------------------------------------------------------
    BenchmarkItem(
        id="bench-31",
        category="coding",
        complexity="high",
        prompt="Write a Python function implementing an async sliding-window rate limiter using Redis sorted sets and pipeline execution.",
        expected_keywords=["async def", "pipeline", "zadd", "zremrangebyscore"],
        required_tier="premium",
    ),
    BenchmarkItem(
        id="bench-32",
        category="coding",
        complexity="high",
        prompt="Explain how to optimize this PostgreSQL query with composite indexing: 'SELECT * FROM audit_logs WHERE tenant_id = $1 AND status_code = 200 ORDER BY created_at DESC LIMIT 50'.",
        expected_keywords=["composite index", "tenant_id", "created_at"],
        required_tier="premium",
    ),
    BenchmarkItem(
        id="bench-33",
        category="coding",
        complexity="high",
        prompt="Implement an asynchronous retry decorator in Python with exponential backoff and randomized full jitter for network exceptions.",
        expected_keywords=["async def", "backoff", "jitter", "retry", "sleep"],
        required_tier="premium",
    ),
    BenchmarkItem(
        id="bench-34",
        category="reasoning",
        complexity="high",
        prompt="Solve the classic river crossing puzzle: A farmer must cross a river with a fox, a goose, and a bag of grain. The boat can only hold the farmer and one item. Fox eats goose, goose eats grain if left alone. Give the exact sequence of trips.",
        expected_keywords=["goose", "fox", "grain", "across"],
        required_tier="premium",
    ),
    BenchmarkItem(
        id="bench-35",
        category="coding",
        complexity="high",
        prompt="Write a Python function to verify a hashed API key using constant-time comparison to prevent timing side-channel attacks.",
        expected_keywords=["compare_digest", "hash", "sha256", "constant"],
        required_tier="premium",
    ),
    BenchmarkItem(
        id="bench-36",
        category="reasoning",
        complexity="high",
        prompt="Explain how a circular deadlock occurs when Thread 1 locks Resource A then waits for B, while Thread 2 locks Resource B then waits for A, and how strict lock ordering eliminates it.",
        expected_keywords=["deadlock", "lock ordering", "hierarchy", "cycle"],
        required_tier="premium",
    ),
    BenchmarkItem(
        id="bench-37",
        category="coding",
        complexity="high",
        prompt="Write an async generator in Python that reads chunks from an upstream stream and yields SSE events formatted as 'data: <json>\\n\\n' followed by 'data: [DONE]\\n\\n'.",
        expected_keywords=["async def", "yield", "data:", "[done]"],
        required_tier="premium",
    ),
    BenchmarkItem(
        id="bench-38",
        category="reasoning",
        complexity="high",
        prompt="Explain the mathematical difference between nearest-rank and linear interpolation when computing p95 and p99 quantiles from discrete latency samples.",
        expected_keywords=["interpolation", "percentile", "rank", "quantile"],
        required_tier="premium",
    ),
    BenchmarkItem(
        id="bench-39",
        category="coding",
        complexity="high",
        prompt="Implement a thread-safe singleton in Python using double-checked locking with an asyncio or threading lock.",
        expected_keywords=["singleton", "lock", "__new__", "instance"],
        required_tier="premium",
    ),
    BenchmarkItem(
        id="bench-40",
        category="reasoning",
        complexity="high",
        prompt="Design a fault-tolerant multi-model fallback state machine with circuit breaking that routes around HTTP 429, 502, and 504 errors while preventing cascading outages.",
        expected_keywords=["circuit breaker", "fallback", "closed", "open", "half_open"],
        required_tier="premium",
    ),
]


def get_benchmark_dataset(sample_size: int = 40, categories: List[str] = None) -> List[BenchmarkItem]:
    """Retrieve benchmark items with optional filtering and sample sizing."""
    items = BENCHMARK_DATASET
    if categories:
        cat_set = {c.lower() for c in categories}
        items = [item for item in items if item.category.lower() in cat_set]
    return items[:sample_size]
