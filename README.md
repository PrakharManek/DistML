# DistML – Distributed ML Inference Engine

A distributed inference engine that horizontally shards ML model layers across multiple worker nodes for scalable, fault-tolerant inference.

## Architecture

```
Client
  │
  ▼
FastAPI Server  ──►  Inference Engine  ──►  Load Balancer
                          │                      │
                    Async Queue          ┌────────┴────────┐
                                    Worker-0          Worker-1 ...
                                   (Layers 0-2)     (Layers 3-5)
                                         │
                                  Redis + PostgreSQL
```

## Features

- **Layer Sharding** — model layers split evenly across N workers; tensors pipelined between shards
- **Dual Routing** — round-robin or least-latency strategy
- **Health Monitoring** — background loop pings every worker every 500ms; unhealthy workers auto-excluded
- **Async Queue** — non-blocking job submission; results polled via `/result/{id}`
- **Worker Metrics** — per-worker latency, success rate, and request counts at `/cluster/stats`
- **Docker Ready** — single `docker-compose up` starts API + Redis + PostgreSQL

## Quick Start

```bash
git clone https://github.com/PrakharManek/DistML.git
cd DistML
pip install -r requirements.txt
uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
```

Or with Docker:

```bash
docker-compose -f docker/docker-compose.yml up --build
```

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness check |
| `GET` | `/cluster/stats` | Worker pool metrics |
| `POST` | `/infer` | Submit inference request |
| `GET` | `/result/{id}` | Poll for result |

### Submit inference

```bash
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{"input_shape": [1, 768], "priority": 1}'
```

```json
{
  "request_id": "a3f1b2c4...",
  "status": "queued",
  "message": "Poll /result/{request_id} for output."
}
```

### Poll result

```bash
curl http://localhost:8000/result/a3f1b2c4...
```

```json
{
  "request_id": "a3f1b2c4...",
  "success": true,
  "output_shape": [1, 768],
  "latency_ms": 12.4,
  "worker_id": "worker-2"
}
```

## Run Tests

```bash
pip install pytest pytest-asyncio
pytest tests/ -v
```

## Tech Stack

| Layer | Tech |
|-------|------|
| Language | Python 3.11 |
| API | FastAPI + Uvicorn |
| ML | PyTorch + NumPy |
| Messaging | gRPC + Redis |
| Database | PostgreSQL |
| Infra | Docker + Compose |

## Project Structure

```
DistML/
├── src/
│   ├── api/server.py              # FastAPI app + lifespan
│   ├── engine/inference_engine.py # Core orchestration + shard pipelining
│   ├── balancer/load_balancer.py  # WorkerPool + routing + health checks
│   └── worker/worker_node.py      # WorkerNode + ModelLayerShard + metrics
├── tests/test_engine.py
├── docker/docker-compose.yml
├── Dockerfile
└── requirements.txt
```

## Author

**Prakhar Manek** — B.Tech IT, IIIT Bhopal

[![GitHub](https://img.shields.io/badge/GitHub-PrakharManek-black?logo=github)](https://github.com/PrakharManek)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-prakhar--manek-blue?logo=linkedin)](https://linkedin.com/in/prakhar-manek-6516473a2)
