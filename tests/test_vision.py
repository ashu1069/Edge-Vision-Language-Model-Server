"""
Tests for VisionModel class.

Covers initialization, TensorRT/ONNX model resolution,
image decoding, and inference with mocked YOLO.
"""

import os
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from app.vision import VisionModel


# ------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------

def test_vision_model_initialization():
    """Test VisionModel can be initialized with default .pt model."""
    with patch("app.vision.YOLO") as mock_yolo:
        mock_model = Mock()
        mock_model.predict = Mock(return_value=[Mock(boxes=[])])
        mock_yolo.return_value = mock_model

        vision = VisionModel()
        assert vision.model is not None
        mock_yolo.assert_called_once_with("yolov8n.pt")


def test_vision_model_custom_model_name():
    """Test initialization with custom model name."""
    with patch("app.vision.YOLO") as mock_yolo:
        mock_model = Mock()
        mock_model.predict = Mock(return_value=[Mock(boxes=[])])
        mock_yolo.return_value = mock_model

        vision = VisionModel(model_name="yolov8s.pt")
        assert vision.model_name == "yolov8s.pt"


def test_vision_model_from_env():
    """Test model name read from YOLO_MODEL env var."""
    with patch("app.vision.YOLO") as mock_yolo, \
         patch.dict(os.environ, {"YOLO_MODEL": "yolov8m.pt"}):
        mock_model = Mock()
        mock_model.predict = Mock(return_value=[Mock(boxes=[])])
        mock_yolo.return_value = mock_model

        vision = VisionModel()
        assert vision.model_name == "yolov8m.pt"


def test_vision_model_half_from_env():
    """Test FP16 flag read from YOLO_HALF env var."""
    with patch("app.vision.YOLO") as mock_yolo, \
         patch.dict(os.environ, {"YOLO_HALF": "true"}):
        mock_model = Mock()
        mock_model.predict = Mock(return_value=[Mock(boxes=[])])
        mock_yolo.return_value = mock_model

        vision = VisionModel()
        assert vision.half is True


# ------------------------------------------------------------------
# Model path resolution
# ------------------------------------------------------------------

def test_resolve_engine_path_directly():
    """Test that .engine file is used directly without export."""
    with patch("app.vision.YOLO") as mock_yolo:
        mock_model = Mock()
        mock_model.predict = Mock(return_value=[Mock(boxes=[])])
        mock_yolo.return_value = mock_model

        vision = VisionModel(model_name="yolov8n.engine")
        # YOLO should be called with the .engine path
        mock_yolo.assert_called_once_with("yolov8n.engine")


def test_resolve_onnx_path_directly():
    """Test that .onnx file is used directly without export."""
    with patch("app.vision.YOLO") as mock_yolo:
        mock_model = Mock()
        mock_model.predict = Mock(return_value=[Mock(boxes=[])])
        mock_yolo.return_value = mock_model

        vision = VisionModel(model_name="yolov8n.onnx")
        mock_yolo.assert_called_once_with("yolov8n.onnx")


def test_export_format_triggers_export_when_engine_missing(tmp_path):
    """Test that YOLO_EXPORT_FORMAT triggers export when no engine exists."""
    pt_file = tmp_path / "model.pt"
    pt_file.touch()
    engine_file = tmp_path / "model.engine"

    with patch("app.vision.YOLO") as mock_yolo:
        # Mock the export
        export_model = Mock()
        export_model.export.return_value = str(engine_file)

        # First YOLO() call is for export, second is for loading
        run_model = Mock()
        run_model.predict = Mock(return_value=[Mock(boxes=[])])
        mock_yolo.side_effect = [export_model, run_model]

        vision = VisionModel(
            model_name=str(pt_file), export_format="engine"
        )
        export_model.export.assert_called_once()


def test_export_format_uses_existing_engine(tmp_path):
    """Test that existing engine file is reused without re-exporting."""
    pt_file = tmp_path / "model.pt"
    pt_file.touch()
    engine_file = tmp_path / "model.engine"
    engine_file.touch()  # Pre-existing engine

    with patch("app.vision.YOLO") as mock_yolo:
        mock_model = Mock()
        mock_model.predict = Mock(return_value=[Mock(boxes=[])])
        mock_yolo.return_value = mock_model

        vision = VisionModel(
            model_name=str(pt_file), export_format="engine"
        )
        # YOLO should be called with the engine path, only once (no export)
        mock_yolo.assert_called_once_with(str(engine_file))


# ------------------------------------------------------------------
# Image decoding
# ------------------------------------------------------------------

def test_decode_image_valid_base64():
    """Test decoding valid base64 image."""
    base64_string = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

    with patch("app.vision.cv2.imdecode") as mock_imdecode:
        mock_imdecode.return_value = np.zeros((1, 1, 3), dtype=np.uint8)
        vision = VisionModel.__new__(VisionModel)
        result = vision.decode_image(base64_string)

        assert result is not None
        assert isinstance(result, np.ndarray)


def test_decode_image_invalid_base64():
    """Test decoding invalid base64."""
    vision = VisionModel.__new__(VisionModel)
    result = vision.decode_image("invalid_base64_string")
    assert result is None


def test_decode_image_empty_string():
    """Test decoding empty string."""
    vision = VisionModel.__new__(VisionModel)
    result = vision.decode_image("")
    assert result is None


# ------------------------------------------------------------------
# Inference
# ------------------------------------------------------------------

def test_predict_with_mock_model(sample_image_base64):
    """Test predict method with mocked YOLO model."""
    with patch("app.vision.YOLO") as mock_yolo:
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
    """Test predict method with mock detections."""
    with patch("app.vision.YOLO") as mock_yolo, \
         patch("app.vision.cv2.imdecode") as mock_imdecode:

        mock_imdecode.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        mock_model = Mock()
        mock_box1 = Mock()
        mock_box1.cls = [0]
        mock_box1.conf = [0.84]
        mock_box1.xywhn = [np.array([0.5, 0.5, 0.2, 0.3])]

        mock_box2 = Mock()
        mock_box2.cls = [2]
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
    """Test predict with invalid image data."""
    vision = VisionModel.__new__(VisionModel)
    result = vision.predict("invalid_base64", conf_threshold=0.5)
    assert "error" in result
    assert result["error"] == "Invalid image data"


def test_predict_confidence_threshold():
    """Test that confidence threshold is passed to YOLO."""
    with patch("app.vision.YOLO") as mock_yolo:
        mock_model = Mock()
        mock_result = Mock()
        mock_result.boxes = []
        mock_result.names = {0: "person"}
        mock_model.predict.return_value = [mock_result]
        mock_yolo.return_value = mock_model

        vision = VisionModel()

        mock_img = np.zeros((100, 100, 3), dtype=np.uint8)
        vision.decode_image = Mock(return_value=mock_img)

        vision.predict("fake_base64", conf_threshold=0.7)

        call_args = mock_model.predict.call_args
        assert call_args is not None, "predict() was not called"
        assert "conf" in call_args.kwargs
        assert call_args.kwargs["conf"] == 0.7


def test_predict_passes_half_flag():
    """Test that half precision flag is forwarded to YOLO predict."""
    with patch("app.vision.YOLO") as mock_yolo:
        mock_model = Mock()
        mock_result = Mock()
        mock_result.boxes = []
        mock_model.predict.return_value = [mock_result]
        mock_yolo.return_value = mock_model

        vision = VisionModel(half=True)

        mock_img = np.zeros((100, 100, 3), dtype=np.uint8)
        vision.decode_image = Mock(return_value=mock_img)

        vision.predict("fake_base64", conf_threshold=0.5)

        call_args = mock_model.predict.call_args
        assert call_args.kwargs["half"] is True


# ------------------------------------------------------------------
# Info
# ------------------------------------------------------------------

def test_get_info():
    """Test get_info returns model metadata."""
    with patch("app.vision.YOLO") as mock_yolo:
        mock_model = Mock()
        mock_model.predict = Mock(return_value=[Mock(boxes=[])])
        mock_yolo.return_value = mock_model

        vision = VisionModel(half=True, export_format="engine")
        info = vision.get_info()

        assert info["half"] is True
        assert info["export_format"] == "engine"
        assert info["imgsz"] == 640
