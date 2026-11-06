"""
gateway/benchmarks
==================
Benchmark evaluation harness comparing All-Premium vs. Routed Gateway.
"""
from gateway.benchmarks.dataset import BENCHMARK_DATASET, BenchmarkItem, get_benchmark_dataset
from gateway.benchmarks.evaluator import BenchmarkEvaluator, EvaluationResult
from gateway.benchmarks.runner import BenchmarkRunner

__all__ = [
    "BenchmarkItem",
    "BENCHMARK_DATASET",
    "get_benchmark_dataset",
    "BenchmarkEvaluator",
    "EvaluationResult",
    "BenchmarkRunner",
]
