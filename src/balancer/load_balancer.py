"""
Load Balancer
Supports round-robin and least-latency routing strategies.
Health-checks workers every 500ms and reroutes traffic on failure.
"""

import asyncio
import logging
import time
from enum import Enum
from typing import Dict, List, Optional

from src.worker.worker_node import WorkerNode, WorkerStatus

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_LATENCY = "least_latency"


class WorkerPool:
    """
    Manages a pool of WorkerNodes with health monitoring and load balancing.
    Health checks run every 500ms in a background task.
    """

    HEALTH_CHECK_INTERVAL = 0.5  # seconds

    def __init__(self, workers: List[WorkerNode], strategy: RoutingStrategy = RoutingStrategy.LEAST_LATENCY):
        self.workers: Dict[int, WorkerNode] = {i: w for i, w in enumerate(workers)}
        self.strategy = strategy
        self._rr_index = 0
        self._health_task: Optional[asyncio.Task] = None
        self._running = False

    def get_worker(self, worker_idx: int) -> WorkerNode:
        """Get a specific worker by index (used by engine for shard routing)."""
        worker = self.workers.get(worker_idx)
        if worker is None or worker.status == WorkerStatus.OFFLINE:
            raise RuntimeError(f"Worker {worker_idx} is unavailable")
        return worker

    def select_worker(self) -> WorkerNode:
        """Select a worker based on the current routing strategy."""
        healthy = [w for w in self.workers.values() if w.status != WorkerStatus.UNHEALTHY and w.status != WorkerStatus.OFFLINE]
        if not healthy:
            raise RuntimeError("No healthy workers available")

        if self.strategy == RoutingStrategy.ROUND_ROBIN:
            worker = healthy[self._rr_index % len(healthy)]
            self._rr_index += 1
            return worker
        else:
            # Least latency: prefer lowest avg response time
            return min(healthy, key=lambda w: w.metrics.avg_latency_ms)

    async def _health_loop(self):
        """Background loop: ping every worker every 500ms."""
        logger.info("[LoadBalancer] Health check loop started (interval=500ms)")
        while self._running:
            for idx, worker in list(self.workers.items()):
                asyncio.create_task(worker.health_check())
            await asyncio.sleep(self.HEALTH_CHECK_INTERVAL)

    async def start(self):
        self._running = True
        self._health_task = asyncio.create_task(self._health_loop())
        logger.info(f"[LoadBalancer] Started with {len(self.workers)} workers | strategy={self.strategy.value}")

    async def stop(self):
        self._running = False
        if self._health_task:
            self._health_task.cancel()
        logger.info("[LoadBalancer] Stopped")

    def cluster_stats(self) -> dict:
        stats = [w.get_stats() for w in self.workers.values()]
        healthy_count = sum(1 for w in self.workers.values() if w.status == WorkerStatus.IDLE)
        return {
            "total_workers": len(self.workers),
            "healthy_workers": healthy_count,
            "strategy": self.strategy.value,
            "workers": stats,
        }
