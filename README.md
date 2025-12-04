# Edge Vision Language Model Server

A production-ready, containerized **edge vision-language inference stack** built with FastAPI, Redis, and YOLOv8 for real-time object detection.

## Features

- **FastAPI REST API** - Clean, async API with automatic OpenAPI documentation
- **YOLOv8 Object Detection** - State-of-the-art vision model for detecting objects in images
- **Redis Queue System** - Reliable job queue with result storage
- **Docker Compose** - One-command deployment with health checks and resource limits
- **Annotation Tools** - Generate annotated images with bounding boxes and JSON outputs
- **Production Ready** - Error handling, graceful shutdown, health checks, and monitoring

## Table of Contents

- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [API Documentation](#api-documentation)
- [Usage Examples](#usage-examples)
- [Development](#development)
- [Testing](#testing)
- [Deployment](#deployment)
- [Roadmap](#roadmap)
- [Contributing](#contributing)

## Architecture

```
┌─────────┐     ┌──────────┐     ┌─────────┐
│ Client  │────▶│   API     │────▶│  Redis  │
└─────────┘     └──────────┘     └────┬────┘
                                      │
                                 ┌────▼────┐
                                 │ Worker  │
                                 │ (YOLOv8)│
                                 └─────────┘
```

### Components

1. **API Service** - FastAPI application handling HTTP requests
2. **Worker Service** - Background processor running YOLOv8 inference
3. **Redis** - Job queue and result storage with persistence

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
# Use the helper script (requires local Python with pillow/opencv)
./inference.sh test.jpg 0.5 "Find all people" --annotate

# Or use curl directly
IMAGE_B64=$(base64 -i test.jpg | tr -d '\n')
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"image_base64\": \"$IMAGE_B64\", \"prompt\": \"Find people\", \"confidence_threshold\": 0.5}" \
  | python3 -m json.tool
```

## API Documentation

### Endpoints

#### `POST /predict`

Submit an image for object detection.

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

**Response (completed):**
```json
{
  "status": "completed",
  "data": {
    "status": "success",
    "vision_result": {
      "detections": [
        {
          "class": "person",
          "confidence": 0.84,
          "box": [0.48, 0.63, 0.78, 0.71]
        }
      ],
      "count": 2,
      "latency_seconds": 0.38
    },
    "vlm_result": "VLM not connected yet"
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

## Usage Examples

### Using the Helper Script

```bash
# Basic inference
./inference.sh test.jpg

# With custom confidence threshold
./inference.sh test.jpg 0.5

# With custom prompt
./inference.sh test.jpg 0.5 "Find all people in this image"

# With annotation (saves annotated image and JSON)
./inference.sh test.jpg 0.5 "Find all people" --annotate
```

### Using cURL

```bash
# Submit image
REQUEST_ID=$(curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"image_base64\": \"$(base64 -i test.jpg | tr -d '\n')\", \"prompt\": \"Find people\", \"confidence_threshold\": 0.5}" \
  | python3 -c "import sys, json; print(json.load(sys.stdin)['request_id'])")

# Get results
curl -s "http://localhost:8000/result/$REQUEST_ID" | python3 -m json.tool
```

### Using Python

```python
import requests
import base64

# Encode image
with open("test.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode('utf-8')

# Submit request
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "image_base64": image_b64,
        "prompt": "Find all people",
        "confidence_threshold": 0.5
    }
)

request_id = response.json()["request_id"]

# Poll for results
import time
while True:
    result = requests.get(f"http://localhost:8000/result/{request_id}")
    data = result.json()
    if data["status"] == "completed":
        print(data["data"])
        break
    time.sleep(1)
```

### Getting Annotated Output

```bash
# Automatically create annotated image and JSON
./inference.sh test.jpg 0.5 "Find all people" --annotate

# Files saved to results/:
# - test_annotated_TIMESTAMP.jpg (image with bounding boxes)
# - test_result_TIMESTAMP.json (complete detection data)
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
uv run pytest tests/test_api.py
```

### Test Structure

```
tests/
├── test_api.py          # API endpoint tests
├── test_vision.py       # Vision model tests
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

# Run API locally (requires Redis)
export REDIS_URL="redis://localhost:6379/0"
export QUEUE_NAME="vlm_queue"
uv run uvicorn app.main:app --reload

# Run worker locally (in another terminal)
export REDIS_URL="redis://localhost:6379/0"
export QUEUE_NAME="vlm_queue"
uv run python -m app.worker
```

### Project Structure

```
.
├── app/
│   ├── main.py          # FastAPI application
│   ├── schemas.py       # Pydantic models
│   ├── vision.py        # YOLOv8 vision model
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

**Key Dependencies:**
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `redis` - Queue and result storage
- `ultralytics` - YOLOv8 models
- `opencv-python-headless` - Image processing
- `pydantic` - Data validation

## Deployment

### Production Considerations

1. **Environment Variables**
   ```bash
   REDIS_URL=redis://redis:6379/0
   REDIS_PASSWORD=your-secure-password
   QUEUE_NAME=vlm_queue
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

See `INFRASTRUCTURE.md` for detailed deployment guide.

## Current Status

### Implemented (Phase 1 & 2)

- [x] FastAPI REST API with async endpoints
- [x] Redis-based job queue system
- [x] YOLOv8 object detection worker
- [x] Health checks and error handling
- [x] Docker Compose setup with resource limits
- [x] Annotation tools for visual results
- [x] Graceful shutdown and connection retry logic
- [x] Comprehensive test suite

### Planned (Phase 3 & 4)

- [ ] Vision-Language Model (VLM) integration
  - [ ] vLLM server setup
  - [ ] VLM worker integration
  - [ ] Multi-modal prompt handling
- [ ] Performance optimization
  - [ ] C++ post-processor
  - [ ] CUDA acceleration
  - [ ] Batch processing

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
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [uv](https://github.com/astral-sh/uv) for blazing-fast package management

## Additional Resources

- [API Documentation](http://localhost:8000/docs) - Interactive Swagger UI
- [Infrastructure Guide](INFRASTRUCTURE.md) - Detailed deployment and scaling guide
- [Annotation Guide](ANNOTATION_GUIDE.md) - Creating annotated outputs

---

**Status**: Production-ready for Phase 1 & 2. Phase 3 & 4 are planned for future releases.
