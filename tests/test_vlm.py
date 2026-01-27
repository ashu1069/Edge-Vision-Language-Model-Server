"""
Tests for VLMModel class.

Tests cover:
- Initialization and configuration
- Image decoding (shape/dtype validation)
- Prompt building with detection context
- Model info and health check data
- Error handling for invalid inputs

Note: Full inference tests require the model to be downloaded.
Use @pytest.mark.slow for tests that load the actual model.
"""

import os
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pytest
from PIL import Image

from app.vlm import VLMModel
from app.device import get_device


class TestGetDevice:
    """Tests for device auto-detection."""
    
    def test_device_override_cuda(self):
        """Test explicit CUDA device override."""
        with patch.dict(os.environ, {"DEVICE": "cuda"}):
            assert get_device() == "cuda"
    
    def test_device_override_mps(self):
        """Test explicit MPS device override."""
        with patch.dict(os.environ, {"DEVICE": "mps"}):
            assert get_device() == "mps"
    
    def test_device_override_cpu(self):
        """Test explicit CPU device override."""
        with patch.dict(os.environ, {"DEVICE": "cpu"}):
            assert get_device() == "cpu"
    
    def test_device_auto_detection_returns_available(self):
        """Test auto-detection returns an available device."""
        with patch.dict(os.environ, {"DEVICE": "auto"}):
            # Since torch is imported inside the function, we test the actual behavior
            # rather than mocking. The function should return a valid device.
            result = get_device()
            assert result in ["cuda", "mps", "cpu"]
            
            # Verify it matches actual hardware availability
            try:
                import torch
                if torch.cuda.is_available():
                    assert result == "cuda"
                elif torch.backends.mps.is_available():
                    assert result == "mps"
                else:
                    assert result == "cpu"
            except ImportError:
                assert result == "cpu"
    
    def test_device_auto_returns_valid_device(self):
        """Test auto-detection returns a valid device string."""
        with patch.dict(os.environ, {"DEVICE": "auto"}):
            device = get_device()
            assert device in ["cuda", "mps", "cpu"]


class TestVLMModelInitialization:
    """Tests for VLMModel initialization."""
    
    def test_init_default_values(self):
        """Test initialization with default values."""
        with patch.dict(os.environ, {}, clear=False):
            # Remove VLM_MODEL if set
            env = os.environ.copy()
            env.pop("VLM_MODEL", None)
            with patch.dict(os.environ, env, clear=True):
                vlm = VLMModel(lazy_load=True)
                assert vlm.model_name == "Qwen/Qwen2-VL-2B-Instruct"
                assert vlm.model is None
                assert vlm.processor is None
                assert vlm._loaded is False
    
    def test_init_custom_model_name(self):
        """Test initialization with custom model name."""
        vlm = VLMModel(model_name="custom/model", lazy_load=True)
        assert vlm.model_name == "custom/model"
    
    def test_init_from_env_var(self):
        """Test initialization reads from environment variable."""
        with patch.dict(os.environ, {"VLM_MODEL": "env/model"}):
            vlm = VLMModel(lazy_load=True)
            assert vlm.model_name == "env/model"
    
    def test_init_explicit_overrides_env(self):
        """Test explicit model_name overrides environment variable."""
        with patch.dict(os.environ, {"VLM_MODEL": "env/model"}):
            vlm = VLMModel(model_name="explicit/model", lazy_load=True)
            assert vlm.model_name == "explicit/model"
    
    def test_init_custom_device(self):
        """Test initialization with custom device."""
        vlm = VLMModel(device="cpu", lazy_load=True)
        assert vlm.device == "cpu"
    
    def test_lazy_load_does_not_load_model(self):
        """Test lazy_load=True does not load the model immediately."""
        vlm = VLMModel(lazy_load=True)
        assert vlm.model is None
        assert vlm.processor is None
        assert vlm._loaded is False


class TestVLMModelDecodeImage:
    """Tests for image decoding functionality."""
    
    @pytest.fixture
    def vlm(self):
        """Create VLMModel instance without loading the model."""
        return VLMModel(lazy_load=True)
    
    def test_decode_valid_base64_png(self, vlm, sample_image_base64):
        """Test decoding valid base64 PNG image."""
        result = vlm.decode_image(sample_image_base64)
        
        assert result is not None
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"
    
    def test_decode_image_converts_to_rgb(self, vlm):
        """Test that RGBA images are converted to RGB."""
        # Create a simple RGBA image and encode it
        import io
        import base64
        
        img = Image.new("RGBA", (10, 10), (255, 0, 0, 128))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        b64_string = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        result = vlm.decode_image(b64_string)
        
        assert result is not None
        assert result.mode == "RGB"
        assert result.size == (10, 10)
    
    def test_decode_invalid_base64(self, vlm):
        """Test decoding invalid base64 returns None."""
        result = vlm.decode_image("not_valid_base64!!!")
        assert result is None
    
    def test_decode_empty_string(self, vlm):
        """Test decoding empty string returns None."""
        result = vlm.decode_image("")
        assert result is None
    
    def test_decode_valid_base64_not_image(self, vlm):
        """Test decoding valid base64 but not an image returns None."""
        import base64
        # Encode some random text
        b64_string = base64.b64encode(b"This is not an image").decode("utf-8")
        result = vlm.decode_image(b64_string)
        assert result is None


class TestVLMModelBuildPrompt:
    """Tests for prompt building with detection context."""
    
    @pytest.fixture
    def vlm(self):
        """Create VLMModel instance without loading the model."""
        return VLMModel(lazy_load=True)
    
    def test_build_prompt_no_context(self, vlm):
        """Test prompt building without detection context."""
        result = vlm._build_prompt("What is in this image?", None)
        assert result == "What is in this image?"
    
    def test_build_prompt_with_empty_detections(self, vlm):
        """Test prompt building with empty detections."""
        context = {"detections": []}
        result = vlm._build_prompt("Describe this scene", context)
        
        assert "No objects were detected" in result
        assert "Describe this scene" in result
    
    def test_build_prompt_with_detections(self, vlm):
        """Test prompt building with detection results."""
        context = {
            "detections": [
                {"class": "person", "confidence": 0.84},
                {"class": "car", "confidence": 0.75},
            ]
        }
        result = vlm._build_prompt("What are they doing?", context)
        
        assert "person" in result
        assert "84%" in result
        assert "car" in result
        assert "75%" in result
        assert "What are they doing?" in result
    
    def test_build_prompt_preserves_user_prompt(self, vlm):
        """Test that user prompt is preserved in output."""
        user_prompt = "Analyze the safety of this scene"
        context = {"detections": [{"class": "pedestrian", "confidence": 0.9}]}
        
        result = vlm._build_prompt(user_prompt, context)
        
        assert user_prompt in result


class TestVLMModelInfo:
    """Tests for model info and health check data."""
    
    def test_get_info_before_load(self):
        """Test get_info returns correct data before model is loaded."""
        vlm = VLMModel(model_name="test/model", device="cpu", lazy_load=True)
        info = vlm.get_info()
        
        assert info["model_name"] == "test/model"
        assert info["device"] == "cpu"
        assert info["loaded"] is False
    
    def test_is_loaded_before_load(self):
        """Test is_loaded returns False before loading."""
        vlm = VLMModel(lazy_load=True)
        assert vlm.is_loaded() is False


class TestVLMModelPredict:
    """Tests for prediction/inference functionality."""
    
    def test_predict_invalid_image(self):
        """Test predict returns error for invalid image."""
        vlm = VLMModel(lazy_load=True)
        # Mock _load_model to avoid actual model loading
        vlm._load_model = Mock()
        vlm._loaded = True
        
        result = vlm.predict("invalid_base64", "What is this?")
        
        assert "error" in result
        assert result["error"] == "Invalid image data"
    
    def test_predict_triggers_lazy_load(self):
        """Test that predict triggers model loading if not loaded."""
        vlm = VLMModel(lazy_load=True)
        vlm._load_model = Mock()
        
        # This will fail after _load_model because model is still None
        # but we can verify _load_model was called
        vlm.decode_image = Mock(return_value=None)  # Force early return
        
        vlm.predict("some_base64", "prompt")
        
        vlm._load_model.assert_called_once()


class TestVLMModelLoadModel:
    """Tests for model loading functionality."""
    
    def test_load_model_import_error(self):
        """Test proper error message when dependencies missing."""
        vlm = VLMModel(lazy_load=True)
        
        with patch.dict("sys.modules", {"transformers": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module")):
                with pytest.raises(ImportError) as exc_info:
                    vlm._load_model()
                
                assert "VLM dependencies not installed" in str(exc_info.value)
    
    def test_load_model_only_loads_once(self):
        """Test that model is only loaded once."""
        vlm = VLMModel(lazy_load=True)
        vlm._loaded = True  # Pretend already loaded
        
        # _load_model should return early
        vlm._load_model()
        
        # Model should still be None (wasn't actually loaded)
        assert vlm.model is None


# Fixtures from conftest.py are available here
class TestVLMModelIntegration:
    """Integration tests that use fixtures."""
    
    def test_with_sample_image(self, sample_image_base64):
        """Test image decoding with sample fixture."""
        vlm = VLMModel(lazy_load=True)
        image = vlm.decode_image(sample_image_base64)
        
        assert image is not None
        assert isinstance(image, Image.Image)
    
    def test_prompt_with_sample_detection(self, sample_detection_result):
        """Test prompt building with sample detection fixture."""
        vlm = VLMModel(lazy_load=True)
        prompt = vlm._build_prompt("What is happening?", sample_detection_result)
        
        assert "person" in prompt
        assert "84%" in prompt
