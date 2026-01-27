# Edge Vision Language Model Server

A production-ready, containerized **edge vision-language inference stack** built with FastAPI, Redis, YOLOv8, and Qwen2-VL for real-time object detection and vision-language reasoning.

## Features

- **FastAPI REST API** - Clean, async API with automatic OpenAPI documentation
- **YOLOv8 Object Detection** - State-of-the-art vision model for detecting objects in images
- **Qwen2-VL Integration** - Vision-language model for image understanding and reasoning
- **Smart Prompt Routing** - Automatically routes requests to YOLO, VLM, or both based on prompt analysis
- **Redis Queue System** - Reliable job queue with result storage
- **Docker Compose** - One-command deployment with health checks and resource limits
- **Multi-Device Support** - Auto-detection of CUDA, MPS (Apple Silicon), and CPU
- **Annotation Tools** - Generate annotated images with bounding boxes and JSON outputs
- **Production Ready** - Error handling, graceful shutdown, health checks, and monitoring

## Table of Contents

- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [API Documentation](#api-documentation)
- [Prompt Routing](#prompt-routing)
- [Usage Examples](#usage-examples)
- [Development](#development)
- [Testing](#testing)
- [Deployment](#deployment)
- [Configuration](#configuration)
- [Contributing](#contributing)

## Architecture

```
┌─────────┐     ┌──────────┐     ┌─────────┐
│ Client  │────▶│   API    │────▶│  Redis  │
└─────────┘     └──────────┘     └────┬────┘
                                      │
                                 ┌────▼────┐
                                 │ Worker  │
                                 └────┬────┘
                                      │
                         ┌────────────┼────────────┐
                         │            │            │
                    ┌────▼────┐  ┌────▼────┐  ┌────▼────┐
                    │  YOLO   │  │   VLM   │  │  Both   │
                    │  Only   │  │  Only   │  │ Models  │
                    └─────────┘  └─────────┘  └─────────┘
```

### Components

1. **API Service** - FastAPI application handling HTTP requests
2. **Worker Service** - Background processor with prompt routing
3. **Prompt Router** - Classifies prompts to determine which model(s) to use
4. **YOLOv8** - Object detection model (fast, always loaded)
5. **Qwen2-VL** - Vision-language model (lazy-loaded, optional)
6. **Redis** - Job queue and result storage with persistence

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.10+ (for local development/testing)

### Running with Docker Compose

```bash
# Clone the repository
git clone <repository-url>
cd Edge-Vision-Language-Model-Server

# Start all services
docker compose up --build

# Services will be available at:
# - API: http://localhost:8000
# - API Docs: http://localhost:8000/docs
# - Health Check: http://localhost:8000/health
```

### Quick Test

```bash
# Detection task
IMAGE_B64=$(base64 -i test.jpg | tr -d '\n')
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"image_base64\": \"$IMAGE_B64\", \"prompt\": \"Find all people\", \"confidence_threshold\": 0.5}"

# VLM reasoning task
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"image_base64\": \"$IMAGE_B64\", \"prompt\": \"Describe what is happening in this scene\"}"

# Combined task
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"image_base64\": \"$IMAGE_B64\", \"prompt\": \"Find all vehicles and describe their behavior\"}"
```

## API Documentation

### Endpoints

#### `POST /predict`

Submit an image for inference. The prompt determines whether to use YOLO, VLM, or both.

**Request:**
```json
{
  "image_base64": "base64_encoded_image_string",
  "prompt": "Find all people in this image",
  "confidence_threshold": 0.5
}
```

**Response:**
```json
{
  "request_id": "uuid-string",
  "status": "queued"
}
```

#### `GET /result/{request_id}`

Get inference results for a request.

**Response (detection only):**
```json
{
  "status": "completed",
  "data": {
    "status": "success",
    "task_type": "detection",
    "vision_result": {
      "detections": [
        {"class": "person", "confidence": 0.84, "box": [0.48, 0.63, 0.2, 0.3]}
      ],
      "count": 1
    },
    "vlm_result": null,
    "latency_seconds": 0.38
  }
}
```

**Response (VLM only):**
```json
{
  "status": "completed",
  "data": {
    "status": "success",
    "task_type": "vlm",
    "vision_result": null,
    "vlm_result": "The image shows a busy street scene with pedestrians crossing...",
    "latency_seconds": 2.45
  }
}
```

**Response (both models):**
```json
{
  "status": "completed",
  "data": {
    "status": "success",
    "task_type": "both",
    "vision_result": {
      "detections": [{"class": "car", "confidence": 0.92, "box": [0.3, 0.4, 0.2, 0.15]}],
      "count": 1
    },
    "vlm_result": "Based on the detected car, it appears to be a sedan traveling at moderate speed...",
    "latency_seconds": 2.85
  }
}
```

#### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "redis": "healthy"
}
```

### Interactive API Documentation

Visit `http://localhost:8000/docs` for interactive Swagger UI documentation.

## Prompt Routing

The system automatically routes requests based on prompt keywords:

| Task Type | Keywords | Example Prompts |
|-----------|----------|-----------------|
| **Detection Only** | find, detect, count, locate, where is, how many | "Find all people", "Count the cars", "Where is the bicycle?" |
| **VLM Only** | describe, explain, why, analyze, safe, risk, summarize | "Describe this scene", "Is this situation safe?", "Explain what is happening" |
| **Both Models** | find and describe, detect and explain, what are they doing | "Find the vehicles and describe their behavior", "What are the people doing?" |

When both detection and VLM keywords are present, or when the prompt is ambiguous, both models are used for comprehensive results.

## Usage Examples

### Using cURL

```bash
# Detection task
REQUEST_ID=$(curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"image_base64\": \"$(base64 -i test.jpg | tr -d '\n')\", \"prompt\": \"Find people\"}" \
  | python3 -c "import sys, json; print(json.load(sys.stdin)['request_id'])")

# Get results
curl -s "http://localhost:8000/result/$REQUEST_ID" | python3 -m json.tool
```

### Using Python

```python
import requests
import base64
import time

# Encode image
with open("test.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode('utf-8')

# Submit request (VLM reasoning)
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "image_base64": image_b64,
        "prompt": "Describe what is happening and identify any safety concerns",
        "confidence_threshold": 0.5
    }
)

request_id = response.json()["request_id"]

# Poll for results
while True:
    result = requests.get(f"http://localhost:8000/result/{request_id}")
    data = result.json()
    if data["status"] == "completed":
        print(f"Task type: {data['data']['task_type']}")
        print(f"Detections: {data['data']['vision_result']}")
        print(f"VLM Response: {data['data']['vlm_result']}")
        break
    time.sleep(1)
```

### Using the Helper Script

```bash
# Basic inference
./inference.sh test.jpg

# With custom confidence threshold
./inference.sh test.jpg 0.5

# With custom prompt
./inference.sh test.jpg 0.5 "Describe this scene"

# With annotation (saves annotated image and JSON)
./inference.sh test.jpg 0.5 "Find all people" --annotate
```

## Testing

### Running Tests

```bash
# Install test dependencies
uv sync --dev

# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=app --cov-report=html

# Run specific test file
uv run pytest tests/test_router.py
```

### Test Structure

```
tests/
├── test_api.py          # API endpoint tests
├── test_vision.py       # Vision model tests
├── test_vlm.py          # VLM model tests
├── test_router.py       # Prompt router tests
├── test_worker.py       # Worker integration tests
└── conftest.py          # Pytest fixtures
```

## Development

### Local Development Setup

```bash
# Install dependencies using uv
uv sync

# Install dev dependencies
uv sync --dev

# Install VLM dependencies (optional, requires more memory)
uv sync --extra vlm

# Run API locally (requires Redis)
export REDIS_URL="redis://localhost:6379/0"
export QUEUE_NAME="vlm_queue"
uv run uvicorn app.main:app --reload

# Run worker locally (in another terminal)
export REDIS_URL="redis://localhost:6379/0"
export VLM_ENABLED=true  # Enable VLM support
uv run python -m app.worker
```

### Project Structure

```
.
├── app/
│   ├── main.py          # FastAPI application
│   ├── schemas.py       # Pydantic models
│   ├── vision.py        # YOLOv8 vision model
│   ├── vlm.py           # Qwen2-VL vision-language model
│   ├── router.py        # Prompt-based task routing
│   ├── device.py        # Device detection utilities
│   └── worker.py        # Background worker
├── tests/               # Test suite
├── results/             # Annotated outputs (gitignored)
├── docker-compose.yml   # Docker Compose configuration
├── Dockerfile           # Container image definition
├── pyproject.toml       # Project dependencies (uv)
├── inference.sh         # Helper script for inference
└── annotate_result.py   # Annotation tool
```

### Dependencies

This project uses [`uv`](https://github.com/astral-sh/uv) for fast, reliable dependency management. All dependencies are defined in `pyproject.toml`.

**Core Dependencies:**
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `redis` - Queue and result storage
- `ultralytics` - YOLOv8 models
- `opencv-python-headless` - Image processing
- `pydantic` - Data validation

**VLM Dependencies (optional):**
- `torch` - PyTorch for model inference
- `transformers` - HuggingFace model loading
- `accelerate` - Model optimization
- `qwen-vl-utils` - Qwen2-VL utilities

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `redis://redis:6379/0` | Redis connection URL |
| `REDIS_PASSWORD` | `changeme` | Redis auth password |
| `QUEUE_NAME` | `vlm_queue` | Redis list name for job queue |
| `VLM_ENABLED` | `true` | Enable/disable VLM (fallback to YOLO-only) |
| `VLM_MODEL` | `Qwen/Qwen2-VL-2B-Instruct` | HuggingFace model ID |
| `LAZY_LOAD_VLM` | `true` | Load VLM on first use vs startup |
| `DEVICE` | `auto` | Force device (auto/cuda/mps/cpu) |

### Resource Requirements

| Component | CPU | Memory | Notes |
|-----------|-----|--------|-------|
| API | 2 cores | 2GB | Handles HTTP requests |
| Worker (YOLO only) | 4 cores | 4GB | Object detection |
| Worker (with VLM) | 4 cores | 8GB+ | VLM needs ~4-6GB for Qwen2-VL-2B |
| Redis | 1 core | 1GB | Job queue and results |

## Deployment

### Production Considerations

1. **Environment Variables**
   ```bash
   REDIS_URL=redis://redis:6379/0
   REDIS_PASSWORD=your-secure-password
   VLM_ENABLED=true
   VLM_MODEL=Qwen/Qwen2-VL-2B-Instruct
   ```

2. **Resource Limits** - Already configured in `docker-compose.yml`

3. **Security**
   - Change default Redis password
   - Use environment variables for secrets
   - Enable HTTPS/TLS
   - Add API authentication

4. **Scaling**
   ```bash
   # Scale workers
   docker compose up --scale worker=3
   ```

5. **Monitoring**
   - Health checks: `GET /health`
   - Logs: `docker compose logs -f`
   - Metrics: Consider adding Prometheus endpoints

## Current Status

### Implemented (Phase 1, 2 & 3)

- [x] FastAPI REST API with async endpoints
- [x] Redis-based job queue system
- [x] YOLOv8 object detection worker
- [x] Health checks and error handling
- [x] Docker Compose setup with resource limits
- [x] Annotation tools for visual results
- [x] Graceful shutdown and connection retry logic
- [x] Comprehensive test suite
- [x] Vision-Language Model (VLM) integration
- [x] Qwen2-VL support via HuggingFace Transformers
- [x] Smart prompt-based routing
- [x] Multi-device support (CUDA/MPS/CPU)

### Planned (Phase 4)

- [ ] Performance optimization
  - [ ] C++ post-processor
  - [ ] Batch processing
- [ ] Streaming responses for VLM
- [ ] API authentication

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting

## License

See [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8
- [Qwen](https://github.com/QwenLM/Qwen2-VL) for Qwen2-VL
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [HuggingFace](https://huggingface.co/) for Transformers
- [uv](https://github.com/astral-sh/uv) for blazing-fast package management

## Additional Resources

- [API Documentation](http://localhost:8000/docs) - Interactive Swagger UI
- [Infrastructure Guide](INFRASTRUCTURE.md) - Detailed deployment and scaling guide
- [Annotation Guide](ANNOTATION_GUIDE.md) - Creating annotated outputs

---

**Status**: Production-ready for Phase 1, 2 & 3. Phase 4 performance optimizations are planned for future releases.
