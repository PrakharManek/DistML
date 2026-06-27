"""
Worker Node
Each worker holds a subset of model layers and executes forward passes for its shard.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class WorkerStatus(Enum):
    IDLE = "idle"
    BUSY = "busy"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"


@dataclass
class WorkerMetrics:
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0.0
    last_health_check: float = field(default_factory=time.time)

    @property
    def avg_latency_ms(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_latency_ms / self.total_requests

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests


class ModelLayerShard:
    """
    Simulates a shard of model layers on a worker.
    In production this would wrap actual PyTorch nn.Module layers.
    """

    def __init__(self, layer_indices: List[int], hidden_dim: int = 768):
        self.layer_indices = layer_indices
        self.hidden_dim = hidden_dim
        # Simulate weight matrices for each assigned layer
        self.weights = {
            idx: np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * 0.02
            for idx in layer_indices
        }

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Run forward pass through assigned layers sequentially."""
        out = x
        for layer_idx in self.layer_indices:
            W = self.weights[layer_idx]
            # Linear transform + ReLU activation (simplified transformer block)
            out = np.maximum(0, out @ W)
        return out


class WorkerNode:
    """
    Represents a single worker node that holds model layer shards.
    Handles forward passes, health checks, and metric tracking.
    """

    def __init__(self, worker_id: str, layer_indices: List[int], hidden_dim: int = 768):
        self.worker_id = worker_id
        self.layer_indices = layer_indices
        self.status = WorkerStatus.IDLE
        self.metrics = WorkerMetrics()
        self.shard = ModelLayerShard(layer_indices, hidden_dim)
        self._lock = asyncio.Lock()

    async def forward(self, input_tensor: np.ndarray, layers: List[int]) -> np.ndarray:
        async with self._lock:
            self.status = WorkerStatus.BUSY
            start = time.time()
            try:
                # Simulate network transfer delay (gRPC serialisation overhead)
                await asyncio.sleep(0.001)
                result = self.shard.forward(input_tensor)
                latency = (time.time() - start) * 1000

                self.metrics.total_requests += 1
                self.metrics.successful_requests += 1
                self.metrics.total_latency_ms += latency
                self.status = WorkerStatus.IDLE
                logger.debug(f"[Worker {self.worker_id}] Forward done | layers={layers} | {latency:.1f}ms")
                return result
            except Exception as e:
                self.metrics.total_requests += 1
                self.metrics.failed_requests += 1
                self.status = WorkerStatus.UNHEALTHY
                logger.error(f"[Worker {self.worker_id}] Forward failed: {e}")
                raise

    async def health_check(self) -> bool:
        """Ping worker; mark unhealthy if unresponsive."""
        try:
            probe = np.ones((1, self.shard.hidden_dim), dtype=np.float32)
            await asyncio.wait_for(self.forward(probe, self.layer_indices), timeout=2.0)
            self.metrics.last_health_check = time.time()
            if self.status == WorkerStatus.UNHEALTHY:
                self.status = WorkerStatus.IDLE
                logger.info(f"[Worker {self.worker_id}] Recovered to IDLE")
            return True
        except Exception:
            self.status = WorkerStatus.UNHEALTHY
            logger.warning(f"[Worker {self.worker_id}] Health check FAILED")
            return False

    def get_stats(self) -> dict:
        return {
            "worker_id": self.worker_id,
            "status": self.status.value,
            "layers": self.layer_indices,
            "total_requests": self.metrics.total_requests,
            "success_rate": round(self.metrics.success_rate, 4),
            "avg_latency_ms": round(self.metrics.avg_latency_ms, 2),
        }
