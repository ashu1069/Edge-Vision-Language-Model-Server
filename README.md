## Edge-Vision-Language-Model-Server

A small, containerized **edge vision-language inference stack** built around:
- **FastAPI** for the public HTTP API
- **Redis** as a lightweight job queue + result store
- A **vision worker** running **YOLOv8** (via `ultralytics`) for object detection

This README describes what is implemented so far, phase by phase.

---

## Phase 1 – The Skeleton (Infrastructure & API) ✅

**Goal:** Stand up the basic infra and API contract, with a queue in the middle and a worker on the side.

What’s implemented:
- **FastAPI service (`api` container)** exposing:
  - `POST /predict` – accepts a JSON payload with:
    - `image_base64`: base64-encoded image
    - `prompt`: free-form text prompt (reserved for future VLM use)
  - Enqueues a job into **Redis** (list `vlm_queue`) and returns:
    - `request_id`: UUID for tracking
    - `status`: `"queued"`
- **Result polling endpoint**:
  - `GET /result/{request_id}` – looks up `result:{request_id}` in Redis:
    - Returns `{"status": "completed", "data": ...}` when a worker has stored a result
    - Returns `{"status": "processing"}` while still in-flight
- **Dockerized setup**:
  - `Dockerfile` building a Python 3.10 image with `uv` and project deps
  - `docker-compose.yml` wiring up:
    - `api` (FastAPI)
    - `worker` (background processor)
    - `redis` (queue + result store)

At the end of Phase 1, the system is fully wired: API → Redis queue → Worker (even if the worker logic is trivial).

---

## Phase 2 – The Eyes (Vision Worker) ✅

**Goal:** Replace the fake vision worker with a real YOLOv8-based detection engine.

What’s implemented:
- **Vision model class (`VisionModel` in `app/vision.py`)**:
  - Loads YOLOv8 (default `yolov8n.pt`) via `ultralytics.YOLO`
  - Decodes base64 images into OpenCV (`cv2`) images
  - Runs inference with configurable `conf_threshold`
  - Returns a JSON-friendly structure:
    - `detections`: list of `{class, confidence, box}` with normalized coordinates
    - `count`: number of detections
- **Worker loop (`app/worker.py`)**:
  - Connects to the same Redis as the API
  - Blocks on `brpop("vlm_queue")` to pull jobs
  - For each job:
    - Runs `VisionModel.predict(...)`
    - Measures latency and adds `latency_seconds`
    - Stores a result at `result:{job_id}` with:
      - `status: "success"`
      - `vision_result`: YOLOv8 output
      - `vlm_result`: placeholder `"VLM not connected yet"`
    - On failure, stores `status: "failed"` with an error message
- **Docker compose wiring**:
  - `worker` service runs `python -m app.worker`
  - Shares image and code with `api` service

At the end of Phase 2, the system performs **real object detection** on submitted images and returns structured results via the `/result/{request_id}` endpoint.

---

## Phase 3 – The Brain (VLM Worker) 🚧

**Goal:** Attach a **Vision-Language Model (VLM)** (e.g., LLaVA, PaliGemma via vLLM) that can consume:
- The **image** and
- The **prompt**

Planned work:
- Stand up a **vLLM** (or similar) server as another service/container
- Extend the worker to:
  - Call the VLM endpoint after vision inference
  - Populate `vlm_result` with the VLM’s response (e.g., captions, answers, grounding)
- Optionally, introduce a second queue for VLM jobs if needed for scaling.

Status: **Not implemented yet** – worker currently returns the placeholder `"VLM not connected yet"` for `vlm_result`.

---

## Phase 4 – The Optimization (C++ Glue) 🚧

**Goal:** Optimize the handoff between vision and language:
- Efficiently post-process YOLO outputs
- Feed them into the VLM with minimal overhead (e.g., via C++/CUDA glue)

Planned work:
- Implement a **C++ post-processor** (with Python bindings) to:
  - Clean and compress detection outputs
  - Prepare structured inputs for the VLM (regions, boxes, labels, etc.)
- Integrate that post-processor into the worker pipeline.

Status: **Design/ideas only** – not started in code yet.

---

## Running the Stack (Dev Workflow)

From the project root:

```bash
docker compose up --build
```

This will start:
- `api` on port **8000**
- `worker` consuming from the `vlm_queue`
- `redis` as the shared queue/store

You can then:
- Visit the FastAPI docs at: `http://localhost:8000/docs`
- Use `POST /predict` with a base64 image and a prompt
- Poll `GET /result/{request_id}` until `status` becomes `"completed"` or `"failed"`.
