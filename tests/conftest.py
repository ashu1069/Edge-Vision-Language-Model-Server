"""
Pytest configuration and fixtures
"""
import pytest
import redis
import json
import os
from unittest.mock import Mock, patch

# Test configuration
TEST_REDIS_HOST = os.getenv("TEST_REDIS_HOST", "localhost")
TEST_REDIS_PORT = int(os.getenv("TEST_REDIS_PORT", "6379"))
TEST_REDIS_DB = int(os.getenv("TEST_REDIS_DB", "1"))  # Use different DB for tests
TEST_REDIS_PASSWORD = os.getenv("TEST_REDIS_PASSWORD", None)

@pytest.fixture
def redis_client():
    """Create a test Redis client"""
    client = redis.Redis(
        host=TEST_REDIS_HOST,
        port=TEST_REDIS_PORT,
        db=TEST_REDIS_DB,
        password=TEST_REDIS_PASSWORD,
        decode_responses=False
    )
    # Clean up test data before and after
    try:
        client.flushdb()
    except:
        pass
    yield client
    # Cleanup after test
    try:
        client.flushdb()
    except:
        pass

@pytest.fixture
def sample_image_base64():
    """Sample base64 encoded image (1x1 red pixel PNG)"""
    # Minimal valid PNG in base64
    return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

@pytest.fixture
def sample_detection_result():
    """Sample detection result structure"""
    return {
        "detections": [
            {
                "class": "person",
                "confidence": 0.84,
                "box": [0.48, 0.63, 0.78, 0.71]
            }
        ],
        "count": 1,
        "latency_seconds": 0.25
    }

@pytest.fixture
def mock_vision_model():
    """Mock VisionModel for testing"""
    with patch('app.vision.VisionModel') as mock:
        instance = Mock()
        instance.predict.return_value = {
            "detections": [{"class": "person", "confidence": 0.84, "box": [0.5, 0.5, 0.2, 0.3]}],
            "count": 1
        }
        mock.return_value = instance
        yield instance

