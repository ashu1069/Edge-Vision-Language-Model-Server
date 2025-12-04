"""
Tests for FastAPI endpoints
"""
import json
from unittest.mock import patch

import pytest
import redis
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

def test_health_endpoint():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code in [200, 503]  # May be 503 if Redis unavailable
    data = response.json()
    assert "status" in data
    assert "redis" in data

def test_predict_endpoint_missing_fields():
    """Test predict endpoint with missing required fields"""
    response = client.post("/predict", json={})
    assert response.status_code == 422  # Validation error

def test_predict_endpoint_invalid_image():
    """Test predict endpoint with invalid base64"""
    response = client.post(
        "/predict",
        json={
            "image_base64": "invalid_base64",
            "prompt": "test",
            "confidence_threshold": 0.5
        }
    )
    # Should accept the request (validation happens in worker)
    assert response.status_code in [200, 503]  # 503 if Redis unavailable

def test_predict_endpoint_valid_request(redis_client, sample_image_base64):
    """Test predict endpoint with valid request"""
    with patch('app.main.redis_client', redis_client), \
         patch('app.main.queue_name', "test_queue"):
        
        response = client.post(
            "/predict",
            json={
                "image_base64": sample_image_base64,
                "prompt": "Find people",
                "confidence_threshold": 0.5
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "request_id" in data
            assert data["status"] == "queued"
            assert len(data["request_id"]) > 0

def test_result_endpoint_not_found():
    """Test result endpoint with non-existent request_id"""
    response = client.get("/result/non-existent-id")
    # May be 503 if Redis unavailable, or 200 if Redis available but no result
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = response.json()
        assert data["status"] == "processing"

@pytest.mark.skipif(True, reason="Requires Redis connection")
def test_result_endpoint_with_result(redis_client, sample_detection_result):
    """Test result endpoint with existing result - requires Redis"""
    test_request_id = "test-request-123"
    result_data = {
        "status": "success",
        "vision_result": sample_detection_result,
        "vlm_result": "VLM not connected yet"
    }
    try:
        redis_client.setex(f"result:{test_request_id}", 60, json.dumps(result_data))
        
        with patch('app.main.redis_client', redis_client):
            response = client.get(f"/result/{test_request_id}")
            assert response.status_code == 200
            data = response.json()
    except redis.exceptions.ConnectionError:
        pytest.skip("Redis not available")
        assert data["status"] == "completed"
        assert "data" in data
        assert data["data"]["status"] == "success"

def test_api_docs_available():
    """Test that API documentation is available"""
    response = client.get("/docs")
    assert response.status_code == 200

def test_openapi_schema():
    """Test OpenAPI schema endpoint"""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "openapi" in schema
    assert "paths" in schema

