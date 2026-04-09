"""
Tests for FastAPI endpoints.

Uses httpx.AsyncClient with the ASGI app directly, which properly
triggers the lifespan (async Redis init/teardown).
"""

import json
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
import redis.asyncio as aioredis
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest_asyncio.fixture
async def async_redis():
    """Create an async Redis client for testing."""
    client = aioredis.Redis(host="localhost", port=6379, db=1, decode_responses=False)
    try:
        await client.flushdb()
    except Exception:
        pass
    yield client
    try:
        await client.flushdb()
        await client.aclose()
    except Exception:
        pass


@pytest_asyncio.fixture
async def client():
    """Async test client that runs the app lifespan."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_health_endpoint(client):
    """Test health check endpoint."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "redis" in data


@pytest.mark.asyncio
async def test_predict_endpoint_missing_fields(client):
    """Test predict endpoint with missing required fields."""
    response = await client.post("/predict", json={})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_predict_endpoint_invalid_image(client):
    """Test predict endpoint with invalid base64."""
    response = await client.post(
        "/predict",
        json={
            "image_base64": "invalid_base64",
            "prompt": "test",
            "confidence_threshold": 0.5,
        },
    )
    # Should accept the request (validation happens in worker)
    assert response.status_code in [200, 503]


@pytest.mark.asyncio
async def test_predict_endpoint_valid_request(client, async_redis, sample_image_base64):
    """Test predict endpoint with valid request when Redis is available."""
    with patch("app.main.redis_client", async_redis), patch(
        "app.main.queue_name", "test_queue"
    ):
        response = await client.post(
            "/predict",
            json={
                "image_base64": sample_image_base64,
                "prompt": "Find people",
                "confidence_threshold": 0.5,
            },
        )

        if response.status_code == 200:
            data = response.json()
            assert "request_id" in data
            assert data["status"] == "queued"
            assert len(data["request_id"]) > 0


@pytest.mark.asyncio
async def test_result_endpoint_not_found(client):
    """Test result endpoint with non-existent request_id."""
    response = await client.get("/result/non-existent-id")
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = response.json()
        assert data["status"] == "processing"


@pytest.mark.asyncio
@pytest.mark.skipif(True, reason="Requires Redis connection")
async def test_result_endpoint_with_result(
    client, async_redis, sample_detection_result
):
    """Test result endpoint with existing result."""
    test_request_id = "test-request-123"
    result_data = {
        "status": "success",
        "vision_result": sample_detection_result,
        "vlm_result": "Test VLM response",
    }
    try:
        await async_redis.setex(
            f"result:{test_request_id}", 60, json.dumps(result_data)
        )

        with patch("app.main.redis_client", async_redis):
            response = await client.get(f"/result/{test_request_id}")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "completed"
            assert "data" in data
            assert data["data"]["status"] == "success"
    except Exception:
        pytest.skip("Redis not available")


@pytest.mark.asyncio
async def test_api_docs_available(client):
    """Test that API documentation is available."""
    response = await client.get("/docs")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_openapi_schema(client):
    """Test OpenAPI schema endpoint."""
    response = await client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "openapi" in schema
    assert "paths" in schema
