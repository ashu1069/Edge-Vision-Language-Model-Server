"""
FastAPI application for Edge VLM Inference Server.

Provides REST API endpoints for image inference using YOLO and VLM models.
Uses async Redis for non-blocking queue operations.
"""

import json
import logging
import os
import uuid
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException

from app.redis_utils import parse_redis_url
from app.schemas import InferenceRequest, InferenceResponse

logger = logging.getLogger(__name__)

redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
queue_name = os.getenv("QUEUE_NAME", "vlm_queue")
redis_password = os.getenv("REDIS_PASSWORD", None)

# Module-level reference, initialized during lifespan
redis_client: aioredis.Redis | None = None


def _build_redis_client() -> aioredis.Redis:
    """Build an async Redis client from environment config."""
    host, port, db = parse_redis_url(redis_url)

    return aioredis.Redis(
        host=host,
        port=port,
        db=db,
        password=redis_password,
        decode_responses=False,
        socket_connect_timeout=5,
        socket_timeout=5,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage async Redis connection lifecycle."""
    global redis_client
    redis_client = _build_redis_client()
    try:
        await redis_client.ping()
        logger.info("Async Redis connection established")
    except Exception as e:
        logger.warning(f"Redis connection failed at startup: {e}")
        await redis_client.aclose()
        redis_client = None
    yield
    if redis_client is not None:
        await redis_client.aclose()
        logger.info("Async Redis connection closed")


app = FastAPI(title="Edge VLM Infra", lifespan=lifespan)


@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest):
    if redis_client is None:
        raise HTTPException(status_code=503, detail="Redis service unavailable")

    request_id = str(uuid.uuid4())
    job_data = {
        "id": request_id,
        "image": request.image_base64,
        "prompt": request.prompt,
        "confidence_threshold": request.confidence_threshold,
    }

    try:
        await redis_client.lpush(queue_name, json.dumps(job_data))
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to queue job: {str(e)}")

    return InferenceResponse(request_id=request_id, status="queued")


@app.get("/result/{request_id}")
async def get_result(request_id: str):
    if redis_client is None:
        raise HTTPException(status_code=503, detail="Redis service unavailable")

    try:
        result = await redis_client.get(f"result:{request_id}")
        if result:
            return {"status": "completed", "data": json.loads(result)}
        else:
            return {"status": "processing"}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve result: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        if redis_client:
            await redis_client.ping()
            redis_status = "healthy"
        else:
            redis_status = "unavailable"
    except Exception:
        redis_status = "unhealthy"

    return {
        "status": "healthy" if redis_status == "healthy" else "degraded",
        "redis": redis_status,
    }
