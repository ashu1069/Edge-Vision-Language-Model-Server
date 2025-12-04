"""
Tests for VisionModel class
"""
import numpy as np
from unittest.mock import Mock, patch

import pytest

from app.vision import VisionModel

def test_vision_model_initialization():
    """Test VisionModel can be initialized"""
    with patch('app.vision.YOLO') as mock_yolo:
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        vision = VisionModel()
        assert vision.model is not None
        mock_yolo.assert_called_once_with('yolov8n.pt')

def test_decode_image_valid_base64():
    """Test decoding valid base64 image"""
    # Create a minimal valid PNG (1x1 red pixel)
    base64_string = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    
    with patch('app.vision.cv2.imdecode') as mock_imdecode:
        mock_imdecode.return_value = np.zeros((1, 1, 3), dtype=np.uint8)
        vision = VisionModel.__new__(VisionModel)  # Create without calling __init__
        result = vision.decode_image(base64_string)
        
        assert result is not None
        assert isinstance(result, np.ndarray)

def test_decode_image_invalid_base64():
    """Test decoding invalid base64"""
    vision = VisionModel.__new__(VisionModel)
    result = vision.decode_image("invalid_base64_string")
    assert result is None

def test_decode_image_empty_string():
    """Test decoding empty string"""
    vision = VisionModel.__new__(VisionModel)
    result = vision.decode_image("")
    assert result is None

def test_predict_with_mock_model(sample_image_base64):
    """Test predict method with mocked YOLO model"""
    with patch('app.vision.YOLO') as mock_yolo:
        # Create mock YOLO model
        mock_model = Mock()
        mock_result = Mock()
        mock_result.boxes = []
        mock_model.predict.return_value = [mock_result]
        mock_yolo.return_value = mock_model
        
        vision = VisionModel()
        result = vision.predict(sample_image_base64, conf_threshold=0.5)
        
        assert "detections" in result
        assert "count" in result
        assert isinstance(result["detections"], list)
        assert result["count"] == 0

def test_predict_with_detections(sample_image_base64):
    """Test predict method with mock detections"""
    with patch('app.vision.YOLO') as mock_yolo, \
         patch('app.vision.cv2.imdecode') as mock_imdecode:
        
        # Mock image
        mock_imdecode.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Create mock YOLO model with detections
        mock_model = Mock()
        mock_box1 = Mock()
        mock_box1.cls = [0]  # person class
        mock_box1.conf = [0.84]
        mock_box1.xywhn = [np.array([0.5, 0.5, 0.2, 0.3])]
        
        mock_box2 = Mock()
        mock_box2.cls = [2]  # car class
        mock_box2.conf = [0.75]
        mock_box2.xywhn = [np.array([0.3, 0.3, 0.1, 0.2])]
        
        mock_result = Mock()
        mock_result.boxes = [mock_box1, mock_box2]
        mock_result.names = {0: "person", 2: "car"}
        
        mock_model.predict.return_value = [mock_result]
        mock_yolo.return_value = mock_model
        
        vision = VisionModel()
        result = vision.predict(sample_image_base64, conf_threshold=0.5)
        
        assert result["count"] == 2
        assert len(result["detections"]) == 2
        assert result["detections"][0]["class"] == "person"
        assert result["detections"][0]["confidence"] == 0.84
        assert result["detections"][1]["class"] == "car"
        assert result["detections"][1]["confidence"] == 0.75

def test_predict_invalid_image():
    """Test predict with invalid image data"""
    vision = VisionModel.__new__(VisionModel)
    result = vision.predict("invalid_base64", conf_threshold=0.5)
    assert "error" in result
    assert result["error"] == "Invalid image data"

def test_predict_confidence_threshold():
    """Test that confidence threshold is passed to YOLO"""
    with patch('app.vision.YOLO') as mock_yolo:
        mock_model = Mock()
        mock_result = Mock()
        mock_result.boxes = []
        mock_result.names = {0: "person"}
        mock_model.predict.return_value = [mock_result]
        mock_yolo.return_value = mock_model
        
        vision = VisionModel()
        
        # Mock decode_image to return a valid image (bypassing base64 decoding)
        mock_img = np.zeros((100, 100, 3), dtype=np.uint8)
        vision.decode_image = Mock(return_value=mock_img)
        
        vision.predict("fake_base64", conf_threshold=0.7)
        
        # Check that conf parameter was used (not conf_threshold)
        call_args = mock_model.predict.call_args
        assert call_args is not None, "predict() was not called"
        assert "conf" in call_args.kwargs
        assert call_args.kwargs["conf"] == 0.7

