# Edge Vision-Language Model Server

A free, open-source inference server that combines **YOLOv8 object detection** with **vision-language model reasoning** for edge deployment. Runs on NVIDIA Jetson, desktop GPUs, Apple Silicon, or CPU.

Point a camera at a scene, ask a question, and get both structured detections and natural-language understanding.

```
"Find all vehicles and describe their behavior"
  → detections: [{class: "car", confidence: 0.92, box: [...]}, ...]
  → vlm_result: "A sedan is stopped at the intersection while a cyclist passes..."
```

## Why This Exists

Most vision-language APIs require cloud access, cost money, and add latency. This server is designed to run **entirely on-premise** on devices as small as a Jetson Nano — no cloud, no API keys, no data leaving your network. Use it for campus safety monitoring, ADAS prototyping, warehouse inspection, or agricultural field analysis.

## Features

- **YOLOv8 Detection** with TensorRT / ONNX export for edge GPUs
- **Any HuggingFace VLM** — Qwen2-VL, SmolVLM, LLaVA, PaliGemma, InternVL, etc.
- **Smart Prompt Routing** — automatically picks YOLO, VLM, or both based on your question
- **INT8 / INT4 Quantization** — run 2B-parameter VLMs in ~1.2 GB VRAM
- **torch.compile()** — graph-level optimization for 10-30% speedup
- **Async Redis Queue** — non-blocking API, reliable job processing
- **Docker Compose** — one-command deployment with health checks
- **Multi-Device** — auto-detects CUDA, MPS (Apple Silicon), CPU

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.10+ (for local development)

### 1. Run with Docker Compose

```bash
git clone https://github.com/your-username/Edge-Vision-Language-Model-Server.git
cd Edge-Vision-Language-Model-Server

# Start all services (API + Worker + Redis)
docker compose up --build

# API available at:
#   http://localhost:8000       — endpoints
#   http://localhost:8000/docs  — interactive Swagger UI
#   http://localhost:8000/health — health check
```

### 2. Send Your First Request

```bash
# Encode an image
IMAGE_B64=$(base64 -i test.jpg | tr -d '\n')       # macOS
# IMAGE_B64=$(base64 -w0 test.jpg)                  # Linux

# Submit a detection request
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"image_base64\": \"$IMAGE_B64\", \"prompt\": \"Find all people\"}" \
  | python3 -m json.tool

# → {"request_id": "abc-123-...", "status": "queued"}

# Poll for results
curl -s http://localhost:8000/result/abc-123-... | python3 -m json.tool
```

### 3. Use the Helper Script

```bash
# Basic inference
./inference.sh test.jpg

# Custom confidence + prompt
./inference.sh test.jpg 0.5 "Describe this scene"

# Save annotated image with bounding boxes
./inference.sh test.jpg 0.5 "Find all people" --annotate
```

## Architecture

```
Client (image + prompt)
    |
    v
FastAPI API  ──>  Redis Queue  ──>  Worker
(async, non-blocking)               |
                              PromptRouter
                            /      |       \
                        YOLO     VLM     Both
                      (detect) (reason) (detect→reason)
                            \      |       /
                              Result ──> Redis ──> Client polls
```

| Component | Role |
|-----------|------|
| **API** ([app/main.py](app/main.py)) | Accepts requests, queues jobs, returns results |
| **Worker** ([app/worker.py](app/worker.py)) | Processes jobs, runs models, stores results |
| **Router** ([app/router.py](app/router.py)) | Classifies prompts → detection / VLM / both |
| **VisionModel** ([app/vision.py](app/vision.py)) | YOLOv8 wrapper with TensorRT/ONNX support |
| **VLMModel** ([app/vlm.py](app/vlm.py)) | Generic HuggingFace VLM with quantization |
| **Redis** | Job queue + result storage (1hr TTL) |

## API Reference

### `POST /predict`

Submit an image for inference.

```json
{
  "image_base64": "<base64-encoded image>",
  "prompt": "Find all people and describe the scene",
  "confidence_threshold": 0.5
}
```

Response:

```json
{ "request_id": "uuid", "status": "queued" }
```

### `GET /result/{request_id}`

Poll for results.

```json
{
  "status": "completed",
  "data": {
    "status": "success",
    "task_type": "both",
    "vision_result": {
      "detections": [
        { "class": "person", "confidence": 0.84, "box": [0.48, 0.63, 0.2, 0.3] }
      ],
      "count": 1
    },
    "vlm_result": "A person is walking across the crosswalk...",
    "latency_seconds": 2.85
  }
}
```

### `GET /health`

Returns `{"status": "healthy", "redis": "healthy"}`.

## Prompt Routing

The router automatically decides which model(s) to use:

| Task | Triggers | Example |
|------|----------|---------|
| **Detection only** | find, detect, count, locate, where is, how many | `"Count the cars"` |
| **VLM only** | describe, explain, analyze, safe, risk, summarize | `"Is this scene safe?"` |
| **Both** | find + describe, ambiguous prompts | `"Find vehicles and describe their behavior"` |

When uncertain, it defaults to running both models (conservative).

## Configuration

All configuration is via environment variables. Set them in your shell, `.env` file, or `docker-compose.yml`.

### Core

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `redis://redis:6379/0` | Redis connection URL |
| `REDIS_PASSWORD` | `changeme` | Redis auth password (**change in production**) |
| `QUEUE_NAME` | `vlm_queue` | Redis queue name |
| `DEVICE` | `auto` | Force device: `auto`, `cuda`, `mps`, `cpu` |

### YOLO (Object Detection)

| Variable | Default | Description |
|----------|---------|-------------|
| `YOLO_MODEL` | `yolov8n.pt` | Model path — `.pt`, `.engine`, or `.onnx` |
| `YOLO_EXPORT_FORMAT` | *(none)* | Auto-export on startup: `engine`, `onnx`, `torchscript`, `openvino` |
| `YOLO_HALF` | `false` | FP16 inference (recommended for TensorRT) |
| `YOLO_IMGSZ` | `640` | Input image size for export |
| `YOLO_WARMUP` | `1` | Warmup inference runs after loading |

### VLM (Vision-Language Model)

| Variable | Default | Description |
|----------|---------|-------------|
| `VLM_ENABLED` | `true` | Enable VLM (set `false` for detection-only mode) |
| `VLM_MODEL` | `Qwen/Qwen2-VL-2B-Instruct` | Any HuggingFace VLM model ID |
| `LAZY_LOAD_VLM` | `true` | Load VLM on first request (saves startup memory) |
| `MODEL_BACKEND` | `auto` | Model loading: `auto`, `transformers`, `qwen` |
| `VLM_QUANTIZATION` | `none` | Weight quantization: `none`, `int8`, `int4` |
| `VLM_COMPILE` | `false` | Apply `torch.compile()` graph optimization |

## Swapping Models

The server works with **any HuggingFace VLM** that supports chat templates:

```bash
# Default (edge-optimized, 2B params, ~4GB VRAM)
VLM_MODEL=Qwen/Qwen2-VL-2B-Instruct

# Lightweight alternative (Apache 2.0 license, ~4GB VRAM)
VLM_MODEL=HuggingFaceTB/SmolVLM-Instruct

# Better reasoning (7B params, ~16GB VRAM)
VLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct

# LLaVA
VLM_MODEL=llava-hf/llava-v1.6-mistral-7b-hf
```

The backend auto-detects Qwen models and uses their optimized loader. All others use the generic `AutoModelForVision2Seq` path.

For YOLO, swap the model size based on your accuracy/speed needs:

```bash
YOLO_MODEL=yolov8n.pt   # Nano  — fastest, lowest accuracy
YOLO_MODEL=yolov8s.pt   # Small — good balance
YOLO_MODEL=yolov8m.pt   # Medium
YOLO_MODEL=yolov8l.pt   # Large — highest accuracy, slowest
```

## Edge Deployment (NVIDIA Jetson)

For Jetson Orin / Nano, use TensorRT + quantization for production performance:

```bash
# .env file for Jetson deployment
DEVICE=cuda
YOLO_MODEL=yolov8n.pt
YOLO_EXPORT_FORMAT=engine    # Auto-exports to TensorRT on first startup
YOLO_HALF=true               # FP16 for TensorRT
VLM_MODEL=Qwen/Qwen2-VL-2B-Instruct
VLM_QUANTIZATION=int4        # 4x memory reduction
VLM_COMPILE=true             # Graph optimization
```

First startup exports the TensorRT engine (~2 min). Subsequent starts load the cached `.engine` file instantly.

**Expected performance on Jetson Orin (8GB):**

| Component | PyTorch FP32 | With Optimizations |
|-----------|-------------|-------------------|
| YOLOv8n | ~30ms | ~5-8ms (TensorRT FP16) |
| Qwen2-VL-2B | ~3-5s, 4GB VRAM | ~1.5-2.5s, ~1.2GB VRAM (INT4) |

## Benchmarks

Measured on Apple M3 Pro (18 GB). See [BENCHMARKS.md](BENCHMARKS.md) for full results.

| Component | Model | Median Latency | Device |
|-----------|-------|---------------|--------|
| **YOLO Detection** | yolov8n.pt | **27.5 ms** | MPS (Apple Silicon) |
| **VLM Inference** | Qwen2-VL-2B | ~38.5 s | MPS (FP32, unoptimized) |

> MPS is not optimized for autoregressive decoding. On CUDA with TensorRT + INT4
> quantization, expect **~5 ms** for YOLO and **~2 s** for VLM on Jetson Orin.

Run benchmarks on your own hardware:

```bash
uv run python benchmark.py --yolo --yolo-runs 50    # YOLO only
uv run python benchmark.py --vlm --vlm-runs 5       # VLM only
uv run python benchmark.py --all                     # Everything
```

## Development

### Local Setup

```bash
# Install dependencies
uv sync --dev

# Install VLM dependencies (optional, needs more memory)
uv sync --dev --extra vlm

# Install quantization support (optional, CUDA only)
uv sync --dev --extra quantization

# Start Redis locally
docker run -d -p 6379:6379 redis:7-alpine

# Terminal 1: API server
export REDIS_URL="redis://localhost:6379/0"
uv run uvicorn app.main:app --reload

# Terminal 2: Worker
export REDIS_URL="redis://localhost:6379/0"
uv run python -m app.worker
```

### Project Structure

```
app/
  main.py          — FastAPI application (async Redis)
  worker.py        — Background job processor
  router.py        — Prompt-based task routing
  vision.py        — YOLOv8 + TensorRT/ONNX wrapper
  vlm.py           — Generic HuggingFace VLM wrapper
  device.py        — Device detection (CUDA/MPS/CPU)
  schemas.py       — Pydantic request/response models
  redis_utils.py   — Shared Redis URL parsing
tests/             — pytest test suite
docker-compose.yml — Full-stack orchestration
Dockerfile         — Container image
inference.sh       — CLI helper for testing
annotate_result.py — Draw bounding boxes on results
```

### Running Tests

```bash
uv run pytest                              # All tests
uv run pytest --cov=app --cov-report=html  # With coverage report
uv run pytest tests/test_router.py         # Single file
```

### Using Python

```python
import requests, base64, time

with open("test.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

# Submit
resp = requests.post("http://localhost:8000/predict", json={
    "image_base64": image_b64,
    "prompt": "Find all people and describe the scene",
    "confidence_threshold": 0.5,
})
request_id = resp.json()["request_id"]

# Poll
while True:
    result = requests.get(f"http://localhost:8000/result/{request_id}").json()
    if result["status"] == "completed":
        print(result["data"]["vlm_result"])
        break
    time.sleep(1)
```

## Scaling

```bash
# Run 3 workers in parallel
docker compose up --scale worker=3

# Workers are stateless — each picks jobs from the shared Redis queue
```

## Resource Requirements

| Component | CPU | Memory | Notes |
|-----------|-----|--------|-------|
| API | 2 cores | 2 GB | HTTP handling only |
| Worker (YOLO only) | 4 cores | 4 GB | Detection tasks |
| Worker (YOLO + VLM) | 4 cores | 8 GB+ | VLM needs ~4 GB (FP32) or ~1.2 GB (INT4) |
| Redis | 1 core | 1 GB | Queue + result storage |

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Add tests for new features
4. Ensure all tests pass (`uv run pytest`)
5. Open a Pull Request

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8
- [Qwen](https://github.com/QwenLM/Qwen2-VL) for Qwen2-VL
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [HuggingFace](https://huggingface.co/) for Transformers
- [uv](https://github.com/astral-sh/uv) for blazing-fast package management
