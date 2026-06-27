"""
DistML API Server
FastAPI endpoints for submitting inference requests and checking results/cluster health.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.balancer.load_balancer import RoutingStrategy, WorkerPool
from src.engine.inference_engine import InferenceEngine
from src.worker.worker_node import WorkerNode

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────
NUM_WORKERS = 4
NUM_LAYERS = 12
HIDDEN_DIM = 768
MODEL_ID = "distml-base"

# ── Globals ──────────────────────────────────────────────────────────────────
worker_pool: Optional[WorkerPool] = None
engine: Optional[InferenceEngine] = None
engine_task: Optional[asyncio.Task] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global worker_pool, engine, engine_task

    # Build workers with evenly distributed layer shards
    layers_per_worker = NUM_LAYERS // NUM_WORKERS
    workers = []
    for i in range(NUM_WORKERS):
        start = i * layers_per_worker
        end = start + layers_per_worker if i < NUM_WORKERS - 1 else NUM_LAYERS
        layers = list(range(start, end))
        workers.append(WorkerNode(worker_id=f"worker-{i}", layer_indices=layers, hidden_dim=HIDDEN_DIM))

    worker_pool = WorkerPool(workers, strategy=RoutingStrategy.LEAST_LATENCY)
    await worker_pool.start()

    engine = InferenceEngine(model_id=MODEL_ID, num_layers=NUM_LAYERS, num_workers=NUM_WORKERS)
    engine_task = asyncio.create_task(engine.run(worker_pool))
    logger.info(f"[API] DistML ready — {NUM_WORKERS} workers, {NUM_LAYERS} layers, dim={HIDDEN_DIM}")

    yield

    engine.stop()
    await worker_pool.stop()
    if engine_task:
        engine_task.cancel()


app = FastAPI(
    title="DistML – Distributed ML Inference Engine",
    description="Horizontally shards model layers across worker nodes for scalable inference.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Schemas ──────────────────────────────────────────────────────────────────

class InferenceRequest(BaseModel):
    input_shape: List[int] = [1, 768]
    priority: int = 1


class InferenceResponse(BaseModel):
    request_id: str
    status: str
    message: str


class ResultResponse(BaseModel):
    request_id: str
    success: bool
    output_shape: Optional[List[int]]
    latency_ms: Optional[float]
    worker_id: Optional[str]
    error: Optional[str]


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_ID, "timestamp": time.time()}


@app.get("/cluster/stats")
async def cluster_stats():
    if worker_pool is None:
        raise HTTPException(status_code=503, detail="Cluster not initialised")
    return worker_pool.cluster_stats()


@app.post("/infer", response_model=InferenceResponse)
async def submit_inference(req: InferenceRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not ready")
    input_tensor = np.random.randn(*req.input_shape).astype(np.float32)
    request_id = await engine.submit(input_tensor, priority=req.priority)
    return InferenceResponse(
        request_id=request_id,
        status="queued",
        message="Request accepted. Poll /result/{request_id} for output.",
    )


@app.get("/result/{request_id}", response_model=ResultResponse)
async def get_result(request_id: str, timeout: float = 30.0):
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not ready")
    result = await engine.get_result(request_id, timeout=timeout)
    if result is None:
        raise HTTPException(status_code=404, detail="Result not found or timed out")
    return ResultResponse(
        request_id=result.request_id,
        success=result.success,
        output_shape=list(result.output.shape) if result.success else None,
        latency_ms=round(result.latency_ms, 2),
        worker_id=result.worker_id,
        error=result.error,
    )


if __name__ == "__main__":
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=False)
