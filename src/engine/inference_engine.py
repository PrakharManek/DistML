"""
DistML Inference Engine
Distributes model layer execution across multiple worker nodes via gRPC.
"""

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    request_id: str
    model_id: str
    input_data: np.ndarray
    priority: int = 1
    timestamp: float = field(default_factory=time.time)


@dataclass
class InferenceResult:
    request_id: str
    model_id: str
    output: np.ndarray
    latency_ms: float
    worker_id: str
    success: bool
    error: Optional[str] = None


class LayerShardConfig:
    """Defines how model layers are split across workers."""

    def __init__(self, total_layers: int, num_workers: int):
        self.total_layers = total_layers
        self.num_workers = num_workers
        self.shards = self._compute_shards()

    def _compute_shards(self) -> Dict[int, List[int]]:
        """Assign layers to workers as evenly as possible."""
        shards: Dict[int, List[int]] = {i: [] for i in range(self.num_workers)}
        for layer_idx in range(self.total_layers):
            worker_idx = layer_idx % self.num_workers
            shards[worker_idx].append(layer_idx)
        return shards

    def get_worker_layers(self, worker_id: int) -> List[int]:
        return self.shards.get(worker_id, [])


class InferenceEngine:
    """
    Core engine that orchestrates distributed inference across worker nodes.
    Handles request queuing, layer sharding, and result aggregation.
    """

    def __init__(self, model_id: str, num_layers: int, num_workers: int):
        self.model_id = model_id
        self.num_layers = num_layers
        self.num_workers = num_workers
        self.shard_config = LayerShardConfig(num_layers, num_workers)
        self._request_queue: asyncio.Queue = asyncio.Queue(maxsize=512)
        self._result_store: Dict[str, InferenceResult] = {}
        self._active = False

    def generate_request_id(self, input_data: np.ndarray) -> str:
        hash_input = f"{self.model_id}-{time.time()}-{input_data.shape}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    async def submit(self, input_data: np.ndarray, priority: int = 1) -> str:
        req_id = self.generate_request_id(input_data)
        request = InferenceRequest(
            request_id=req_id,
            model_id=self.model_id,
            input_data=input_data,
            priority=priority,
        )
        await self._request_queue.put(request)
        logger.info(f"[Engine] Queued request {req_id} | queue_size={self._request_queue.qsize()}")
        return req_id

    async def get_result(self, request_id: str, timeout: float = 30.0) -> Optional[InferenceResult]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if request_id in self._result_store:
                return self._result_store.pop(request_id)
            await asyncio.sleep(0.05)
        return None

    async def _process_request(self, request: InferenceRequest, worker_pool) -> InferenceResult:
        start = time.time()
        try:
            # Pipeline through shards: each worker handles its assigned layers
            intermediate = request.input_data
            last_worker_id = "none"

            for worker_idx in range(self.num_workers):
                layers = self.shard_config.get_worker_layers(worker_idx)
                if not layers:
                    continue
                worker = worker_pool.get_worker(worker_idx)
                intermediate = await worker.forward(intermediate, layers)
                last_worker_id = worker.worker_id

            latency_ms = (time.time() - start) * 1000
            return InferenceResult(
                request_id=request.request_id,
                model_id=request.model_id,
                output=intermediate,
                latency_ms=latency_ms,
                worker_id=last_worker_id,
                success=True,
            )
        except Exception as e:
            logger.error(f"[Engine] Request {request.request_id} failed: {e}")
            return InferenceResult(
                request_id=request.request_id,
                model_id=request.model_id,
                output=np.array([]),
                latency_ms=(time.time() - start) * 1000,
                worker_id="error",
                success=False,
                error=str(e),
            )

    async def run(self, worker_pool):
        self._active = True
        logger.info(f"[Engine] Started for model={self.model_id}, workers={self.num_workers}")
        while self._active:
            try:
                request = await asyncio.wait_for(self._request_queue.get(), timeout=1.0)
                result = await self._process_request(request, worker_pool)
                self._result_store[result.request_id] = result
                self._request_queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"[Engine] Unexpected error: {e}")

    def stop(self):
        self._active = False
        logger.info("[Engine] Shutting down.")
