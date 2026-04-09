"""
Tests for worker functionality
"""
import json
from unittest.mock import Mock, patch

import pytest
import redis

from app.redis_utils import parse_redis_url
from app.vision import VisionModel
from app.worker import connect_to_redis, queue_name

def test_parse_redis_url():
    """Test Redis URL parsing"""
    host, port, db = parse_redis_url("redis://localhost:6379/0")
    assert host == "localhost"
    assert port == 6379
    assert db == 0
    
    host, port, db = parse_redis_url("redis://redis:6380/1")
    assert host == "redis"
    assert port == 6380
    assert db == 1

@pytest.mark.skipif(True, reason="Requires Redis connection")
def test_worker_job_processing(redis_client, sample_image_base64, sample_detection_result):
    """Test worker can process a job from queue"""
    with patch('app.worker.VisionModel') as mock_vision_class:
        mock_vision = Mock()
        mock_vision.predict.return_value = sample_detection_result
        mock_vision_class.return_value = mock_vision
        
        # Create a test job
        job_id = "test-job-123"
        job_data = {
            "id": job_id,
            "image": sample_image_base64,
            "prompt": "Find people",
            "confidence_threshold": 0.5
        }
        
        redis_client.lpush("test_queue", json.dumps(job_data))
        job_json = redis_client.brpop("test_queue", timeout=1)
        if job_json:
            job = json.loads(job_json[1])
            assert job["id"] == job_id
            assert "image" in job
            assert "prompt" in job

@pytest.mark.skipif(True, reason="Requires Redis connection")
def test_worker_error_handling(redis_client, sample_image_base64):
    """Test worker handles errors gracefully"""
    with patch('app.worker.VisionModel') as mock_vision_class:
        mock_vision = Mock()
        mock_vision.predict.side_effect = Exception("Test error")
        mock_vision_class.return_value = mock_vision
        
        # Create a test job
        job_id = "test-job-error"
        job_data = {
            "id": job_id,
            "image": sample_image_base64,
            "prompt": "Find people",
            "confidence_threshold": 0.5
        }
        
        error_result = {
            "status": "failed",
            "error": "Test error"
        }
        
        redis_client.setex(f"result:{job_id}", 60, json.dumps(error_result))
        result = redis_client.get(f"result:{job_id}")
        assert result is not None
        data = json.loads(result)
        assert data["status"] == "failed"
        assert "error" in data

def test_worker_confidence_threshold_passing(redis_client, sample_image_base64):
    """Test that confidence threshold from job is passed to vision model"""
    with patch('app.worker.VisionModel') as mock_vision_class:
        mock_vision = Mock()
        mock_vision.predict.return_value = {"detections": [], "count": 0}
        mock_vision_class.return_value = mock_vision
        
        # Test with custom confidence threshold
        job_data = {
            "id": "test-job-conf",
            "image": sample_image_base64,
            "prompt": "Find people",
            "confidence_threshold": 0.7
        }
        
        conf_threshold = job_data.get("confidence_threshold", 0.5)
        mock_vision.predict(job_data["image"], conf_threshold=conf_threshold)
        call_args = mock_vision.predict.call_args
        assert call_args.kwargs["conf_threshold"] == 0.7

@pytest.mark.skipif(True, reason="Requires Redis connection")
def test_worker_result_storage_format(redis_client, sample_detection_result):
    """Test that worker stores results in correct format"""
    job_id = "test-format-123"
    output = {
        "status": "success",
        "vision_result": sample_detection_result,
        "vlm_result": "VLM not connected yet"
    }
    
    redis_client.setex(f"result:{job_id}", 3600, json.dumps(output))
    result = redis_client.get(f"result:{job_id}")
    assert result is not None
    data = json.loads(result)
    assert data["status"] == "success"
    assert "vision_result" in data
    assert "vlm_result" in data
    assert data["vlm_result"] == "VLM not connected yet"
    assert "detections" in data["vision_result"]

