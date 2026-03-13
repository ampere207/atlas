import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RetrievalMetric:
    """Single retrieval metric event."""

    timestamp: str
    query: str
    strategy: str
    retriever_source: str
    latency_ms: float
    documents_returned: int
    cache_hit: bool
    reranked: bool
    final_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class RetrievalMetrics:
    """Tracks retrieval performance metrics."""

    def __init__(self, metrics_file: str | None = None) -> None:
        """
        Initialize metrics tracker.

        Args:
            metrics_file: Path to save metrics JSON log (optional)
        """
        self.metrics_file = metrics_file
        self._metrics: list[RetrievalMetric] = []
        self._lock = Lock()
        self._strategy_counts: dict[str, int] = {}
        self._source_counts: dict[str, int] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_latency = 0.0
        self._latency_count = 0

    def record_retrieval(
        self,
        query: str,
        strategy: str,
        retriever_source: str,
        latency_ms: float,
        documents_returned: int,
        cache_hit: bool = False,
        reranked: bool = False,
        final_score: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a retrieval event."""
        metric = RetrievalMetric(
            timestamp=datetime.now().isoformat(),
            query=query,
            strategy=strategy,
            retriever_source=retriever_source,
            latency_ms=latency_ms,
            documents_returned=documents_returned,
            cache_hit=cache_hit,
            reranked=reranked,
            final_score=final_score,
            metadata=metadata or {},
        )

        with self._lock:
            self._metrics.append(metric)

            # Update counters
            self._strategy_counts[strategy] = self._strategy_counts.get(strategy, 0) + 1
            self._source_counts[retriever_source] = (
                self._source_counts.get(retriever_source, 0) + 1
            )

            if cache_hit:
                self._cache_hits += 1
            else:
                self._cache_misses += 1

            self._total_latency += latency_ms
            self._latency_count += 1

        # Optionally persist to file
        if self.metrics_file:
            self._save_metric(metric)

        logger.debug(
            f"Recorded metric: {strategy}/{retriever_source} in {latency_ms:.1f}ms"
        )

    def get_strategy_usage(self) -> dict[str, int]:
        """Get retrieval strategy usage counts."""
        with self._lock:
            return dict(self._strategy_counts)

    def get_source_usage(self) -> dict[str, int]:
        """Get retriever source usage counts."""
        with self._lock:
            return dict(self._source_counts)

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache performance statistics."""
        with self._lock:
            total = self._cache_hits + self._cache_misses
            hit_rate = (
                self._cache_hits / total if total > 0 else 0.0
            )
            return {
                "hits": self._cache_hits,
                "misses": self._cache_misses,
                "total": total,
                "hit_rate": hit_rate,
            }

    def get_latency_stats(self) -> dict[str, Any]:
        """Get latency statistics."""
        if not self._metrics:
            return {
                "avg_latency_ms": 0.0,
                "min_latency_ms": 0.0,
                "max_latency_ms": 0.0,
                "total_queries": 0,
            }

        with self._lock:
            latencies = [m.latency_ms for m in self._metrics]
            avg = sum(latencies) / len(latencies) if latencies else 0.0
            return {
                "avg_latency_ms": avg,
                "min_latency_ms": min(latencies),
                "max_latency_ms": max(latencies),
                "total_queries": len(latencies),
            }

    def get_summary(self) -> dict[str, Any]:
        """Get comprehensive metrics summary."""
        return {
            "timestamp": datetime.now().isoformat(),
            "total_queries": len(self._metrics),
            "strategy_usage": self.get_strategy_usage(),
            "source_usage": self.get_source_usage(),
            "cache_stats": self.get_cache_stats(),
            "latency_stats": self.get_latency_stats(),
        }

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._metrics.clear()
            self._strategy_counts.clear()
            self._source_counts.clear()
            self._cache_hits = 0
            self._cache_misses = 0
            self._total_latency = 0.0
            self._latency_count = 0

    def _save_metric(self, metric: RetrievalMetric) -> None:
        """Save metric to file."""
        try:
            if self.metrics_file:
                path = Path(self.metrics_file)
                path.parent.mkdir(parents=True, exist_ok=True)

                with open(path, "a") as f:
                    f.write(json.dumps(asdict(metric)) + "\n")
        except Exception as e:
            logger.error(f"Failed to save metric: {e}")


# Global metrics instance
_global_metrics: RetrievalMetrics | None = None


def get_metrics() -> RetrievalMetrics:
    """Get or create global metrics instance."""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = RetrievalMetrics(
            metrics_file="logs/retrieval_metrics.jsonl"
        )
    return _global_metrics
