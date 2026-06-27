"""
Tests for DistML inference engine, load balancer, and worker nodes.
"""

import asyncio

import numpy as np
import pytest

from src.balancer.load_balancer import RoutingStrategy, WorkerPool
from src.engine.inference_engine import InferenceEngine, LayerShardConfig
from src.worker.worker_node import WorkerNode, WorkerStatus


# ── LayerShardConfig ──────────────────────────────────────────────────────────

def test_shard_config_distributes_all_layers():
    cfg = LayerShardConfig(total_layers=12, num_workers=4)
    all_layers = [l for layers in cfg.shards.values() for l in layers]
    assert sorted(all_layers) == list(range(12))


def test_shard_config_single_worker():
    cfg = LayerShardConfig(total_layers=6, num_workers=1)
    assert cfg.get_worker_layers(0) == [0, 1, 2, 3, 4, 5]


def test_shard_config_more_workers_than_layers():
    cfg = LayerShardConfig(total_layers=3, num_workers=6)
    all_layers = [l for layers in cfg.shards.values() for l in layers]
    assert sorted(all_layers) == list(range(3))


# ── WorkerNode ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_worker_forward_returns_correct_shape():
    worker = WorkerNode(worker_id="w0", layer_indices=[0, 1, 2], hidden_dim=64)
    x = np.ones((1, 64), dtype=np.float32)
    out = await worker.forward(x, [0, 1, 2])
    assert out.shape == (1, 64)


@pytest.mark.asyncio
async def test_worker_metrics_update_on_forward():
    worker = WorkerNode(worker_id="w0", layer_indices=[0], hidden_dim=64)
    x = np.ones((1, 64), dtype=np.float32)
    await worker.forward(x, [0])
    assert worker.metrics.total_requests == 1
    assert worker.metrics.successful_requests == 1
    assert worker.metrics.avg_latency_ms > 0


@pytest.mark.asyncio
async def test_worker_health_check_passes():
    worker = WorkerNode(worker_id="w0", layer_indices=[0], hidden_dim=64)
    healthy = await worker.health_check()
    assert healthy is True
    assert worker.status == WorkerStatus.IDLE


# ── WorkerPool ────────────────────────────────────────────────────────────────

def make_pool(n=4, strategy=RoutingStrategy.ROUND_ROBIN) -> WorkerPool:
    workers = [
        WorkerNode(worker_id=f"w{i}", layer_indices=[i], hidden_dim=64)
        for i in range(n)
    ]
    return WorkerPool(workers, strategy=strategy)


@pytest.mark.asyncio
async def test_pool_round_robin_cycles():
    pool = make_pool(3, RoutingStrategy.ROUND_ROBIN)
    ids = [pool.select_worker().worker_id for _ in range(6)]
    assert ids == ["w0", "w1", "w2", "w0", "w1", "w2"]


@pytest.mark.asyncio
async def test_pool_cluster_stats_structure():
    pool = make_pool(2)
    stats = pool.cluster_stats()
    assert stats["total_workers"] == 2
    assert "workers" in stats
    assert len(stats["workers"]) == 2


# ── InferenceEngine ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_engine_submit_and_get_result():
    workers = [WorkerNode(worker_id=f"w{i}", layer_indices=[i, i + 4], hidden_dim=64) for i in range(4)]
    pool = WorkerPool(workers, strategy=RoutingStrategy.ROUND_ROBIN)
    await pool.start()

    engine = InferenceEngine(model_id="test-model", num_layers=8, num_workers=4)
    engine_task = asyncio.create_task(engine.run(pool))

    x = np.ones((1, 64), dtype=np.float32)
    req_id = await engine.submit(x)
    assert len(req_id) == 16  # sha256 hex prefix

    result = await engine.get_result(req_id, timeout=10.0)
    assert result is not None
    assert result.success is True
    assert result.output.shape == (1, 64)

    engine.stop()
    engine_task.cancel()
    await pool.stop()


@pytest.mark.asyncio
async def test_engine_result_timeout_returns_none():
    workers = [WorkerNode(worker_id="w0", layer_indices=[0], hidden_dim=64)]
    pool = WorkerPool(workers)
    engine = InferenceEngine(model_id="test", num_layers=1, num_workers=1)
    # Don't start the engine — result will never arrive
    result = await engine.get_result("nonexistent-id", timeout=0.1)
    assert result is None
    await pool.stop()
