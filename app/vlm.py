"""
Vision-Language Model wrapper for Qwen2-VL.

Provides text generation capabilities given an image and prompt.
Designed to complement YOLOv8 detection with reasoning/description tasks.
"""

import base64
import io
import logging
import os
from typing import Optional

from PIL import Image

from app.device import get_device

logger = logging.getLogger(__name__)


class VLMModel:
    """
    Vision-Language Model wrapper for Qwen2-VL.
    
    Loads the model lazily or eagerly based on configuration.
    Supports CUDA, MPS (Apple Silicon), and CPU inference.
    
    Attributes:
        model_name: HuggingFace model ID
        device: Target device for inference
        model: The loaded model (None if not yet loaded)
        processor: The model's processor/tokenizer
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        lazy_load: bool = True
    ):
        """
        Initialize the VLM wrapper.
        
        Args:
            model_name: HuggingFace model ID. Defaults to env var VLM_MODEL
                        or 'Qwen/Qwen2-VL-2B-Instruct'.
            device: Target device. Defaults to auto-detection.
            lazy_load: If True, defer model loading until first inference.
        """
        self.model_name = model_name or os.getenv(
            "VLM_MODEL", "Qwen/Qwen2-VL-2B-Instruct"
        )
        self.device = device or get_device()
        self.model = None
        self.processor = None
        self._loaded = False
        
        logger.info(f"VLMModel initialized: model={self.model_name}, device={self.device}")
        
        if not lazy_load:
            self._load_model()
    
    def _load_model(self) -> None:
        """
        Load the model and processor from HuggingFace.
        
        Raises:
            ImportError: If required packages are not installed.
            Exception: If model loading fails.
        """
        if self._loaded:
            return
        
        logger.info(f"Loading VLM: {self.model_name} on {self.device}...")
        
        try:
            import torch
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        except ImportError as e:
            raise ImportError(
                "VLM dependencies not installed. Run: "
                "uv add transformers torch accelerate qwen-vl-utils"
            ) from e
        
        # Determine torch dtype based on device
        if self.device == "cuda":
            torch_dtype = torch.bfloat16
        elif self.device == "mps":
            # MPS has limited bfloat16 support, use float16
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
        
        try:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
            )
            
            # Move to device if not using device_map
            if self.device != "cuda":
                self.model = self.model.to(self.device)
            
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            
            self._loaded = True
            logger.info(f"VLM loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load VLM: {e}")
            raise
    
    def decode_image(self, base64_string: str) -> Optional[Image.Image]:
        """
        Convert base64 string to PIL Image.
        
        Args:
            base64_string: Base64-encoded image data.
            
        Returns:
            PIL Image in RGB format, or None if decoding fails.
        """
        try:
            img_data = base64.b64decode(base64_string)
            img = Image.open(io.BytesIO(img_data))
            # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
            if img.mode != "RGB":
                img = img.convert("RGB")
            return img
        except Exception as e:
            logger.error(f"Error decoding image: {e}")
            return None
    
    def predict(
        self,
        image_base64: str,
        prompt: str,
        max_new_tokens: int = 256,
        detection_context: Optional[dict] = None,
    ) -> dict:
        """
        Generate text response given an image and prompt.
        
        Args:
            image_base64: Base64-encoded image.
            prompt: User's text prompt/question.
            max_new_tokens: Maximum tokens to generate.
            detection_context: Optional YOLO detection results to include
                              in the prompt for grounded reasoning.
        
        Returns:
            dict with keys:
                - 'response': Generated text
                - 'error': Error message if failed
        """
        # Ensure model is loaded
        if not self._loaded:
            self._load_model()
        
        # Decode image
        image = self.decode_image(image_base64)
        if image is None:
            return {"error": "Invalid image data"}
        
        # Build the prompt with optional detection context
        system_prompt = self._build_prompt(prompt, detection_context)
        
        try:
            import torch
            
            # Prepare messages in Qwen2-VL format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": system_prompt},
                    ],
                }
            ]
            
            # Process inputs
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            inputs = self.processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt",
            )
            
            # Move inputs to device
            inputs = inputs.to(self.device)
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Deterministic for reproducibility
                )
            
            # Decode only the new tokens
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            response = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            
            return {"response": response.strip()}
            
        except Exception as e:
            logger.error(f"VLM inference failed: {e}")
            return {"error": str(e)}
    
    def _build_prompt(
        self,
        user_prompt: str,
        detection_context: Optional[dict] = None,
    ) -> str:
        """
        Build the full prompt, optionally including detection context.
        
        Args:
            user_prompt: The user's original prompt.
            detection_context: YOLO detection results (optional).
            
        Returns:
            Formatted prompt string.
        """
        if detection_context is None:
            return user_prompt
        
        # Format detection results for context
        detections = detection_context.get("detections", [])
        if not detections:
            context = "No objects were detected in this image."
        else:
            detection_strs = []
            for det in detections:
                cls = det.get("class", "unknown")
                conf = det.get("confidence", 0)
                detection_strs.append(f"- {cls} (confidence: {conf:.0%})")
            context = "Detected objects:\n" + "\n".join(detection_strs)
        
        return f"{context}\n\nUser question: {user_prompt}"
    
    def is_loaded(self) -> bool:
        """Check if the model is currently loaded."""
        return self._loaded
    
    def get_info(self) -> dict:
        """
        Get model information for health checks.
        
        Returns:
            dict with model name, device, and loaded status.
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "loaded": self._loaded,
        }
