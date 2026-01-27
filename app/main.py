"""
FastAPI application for Edge VLM Inference Server.

Provides REST API endpoints for image inference using YOLO and VLM models.
"""

import json
import logging
import os
import uuid

import redis
from fastapi import FastAPI, HTTPException

from app.schemas import InferenceRequest, InferenceResponse

logger = logging.getLogger(__name__)

app = FastAPI(title="Edge VLM Infra")

redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
queue_name = os.getenv("QUEUE_NAME", "vlm_queue")
redis_password = os.getenv("REDIS_PASSWORD", None)

try:
    if redis_url.startswith("redis://"):
        parts = redis_url.replace("redis://", "").split("/")
        host_port = parts[0].split(":")
        redis_host = host_port[0] if len(host_port) > 0 else "redis"
        redis_port = int(host_port[1]) if len(host_port) > 1 else 6379
        redis_db = int(parts[1]) if len(parts) > 1 else 0
    else:
        redis_host = "redis"
        redis_port = 6379
        redis_db = 0

    redis_client = redis.Redis(
        host=redis_host,
        port=redis_port,
        db=redis_db,
        password=redis_password,
        decode_responses=False,
        socket_connect_timeout=5,
        socket_timeout=5,
    )
    redis_client.ping()
except Exception as e:
    logger.warning(f"Redis connection failed: {e}")
    redis_client = None

@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest):
    if redis_client is None:
        raise HTTPException(status_code=503, detail="Redis service unavailable")
    
    request_id = str(uuid.uuid4())
    job_data = {
        "id": request_id,
        "image": request.image_base64,
        "prompt": request.prompt,
        "confidence_threshold": request.confidence_threshold
    }

    try:
        redis_client.lpush(queue_name, json.dumps(job_data))
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to queue job: {str(e)}")

    return InferenceResponse(
        request_id=request_id,
        status="queued"
    )

@app.get("/result/{request_id}")
async def get_result(request_id: str):
    if redis_client is None:
        raise HTTPException(status_code=503, detail="Redis service unavailable")
    
    try:
        result = redis_client.get(f"result:{request_id}")
        if result:
            return {"status": "completed", "data": json.loads(result)}
        else:
            return {"status": "processing"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve result: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        if redis_client:
            redis_client.ping()
            redis_status = "healthy"
        else:
            redis_status = "unavailable"
    except Exception:
        redis_status = "unhealthy"
    
    return {
        "status": "healthy" if redis_status == "healthy" else "degraded",
        "redis": redis_status
    }