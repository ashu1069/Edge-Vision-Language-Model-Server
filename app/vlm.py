"""
Vision-Language Model wrapper with generic HuggingFace backend.

Supports any HuggingFace VLM that follows the standard chat template
format (Qwen2-VL, SmolVLM, InternVL, LLaVA-Next, PaliGemma, etc.).

Auto-detects Qwen models and uses their optimized class; falls back
to AutoModelForVision2Seq for everything else.

Optimization features:
  - INT8 / INT4 quantization via bitsandbytes (ideal for edge GPUs)
  - torch.compile() graph optimization (PyTorch 2.0+)
  - Automatic GPU memory cleanup between inferences
"""

import base64
import io
import logging
import os
from enum import Enum
from typing import Optional

from PIL import Image

from app.device import get_device

logger = logging.getLogger(__name__)


class QuantizationMode(str, Enum):
    """Supported weight quantization modes."""

    NONE = "none"
    INT8 = "int8"   # ~2x memory reduction, minimal quality loss
    INT4 = "int4"   # ~4x memory reduction, some quality loss


class ModelBackend(str, Enum):
    """Supported model loading backends."""

    AUTO = "auto"  # Auto-detect from model name
    TRANSFORMERS = "transformers"  # Generic AutoModel path
    QWEN = "qwen"  # Qwen-specific optimized path


def _detect_backend(model_name: str) -> ModelBackend:
    """Infer the best backend from the model name."""
    name_lower = model_name.lower()
    if "qwen" in name_lower and "vl" in name_lower:
        return ModelBackend.QWEN
    return ModelBackend.TRANSFORMERS


class VLMModel:
    """
    Vision-Language Model wrapper supporting multiple HuggingFace models.

    Loads the model lazily or eagerly based on configuration.
    Supports CUDA, MPS (Apple Silicon), and CPU inference.

    Attributes:
        model_name: HuggingFace model ID
        device: Target device for inference
        backend: Which loading strategy to use
        model: The loaded model (None if not yet loaded)
        processor: The model's processor/tokenizer
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        lazy_load: bool = True,
        backend: Optional[str] = None,
        quantization: Optional[str] = None,
        compile_model: Optional[bool] = None,
    ):
        """
        Initialize the VLM wrapper.

        Args:
            model_name: HuggingFace model ID. Defaults to env var VLM_MODEL
                        or 'Qwen/Qwen2-VL-2B-Instruct'.
            device: Target device. Defaults to auto-detection.
            lazy_load: If True, defer model loading until first inference.
            backend: Model backend ('auto', 'transformers', 'qwen').
                     Defaults to env var MODEL_BACKEND or 'auto'.
            quantization: Weight quantization mode ('none', 'int8', 'int4').
                          Defaults to env var VLM_QUANTIZATION or 'none'.
                          Requires bitsandbytes and a CUDA device.
            compile_model: Apply torch.compile() after loading (PyTorch 2.0+).
                           Defaults to env var VLM_COMPILE or False.
        """
        self.model_name = model_name or os.getenv(
            "VLM_MODEL", "Qwen/Qwen2-VL-2B-Instruct"
        )
        self.device = device or get_device()
        self.model = None
        self.processor = None
        self._loaded = False

        backend_str = backend or os.getenv("MODEL_BACKEND", "auto")
        if backend_str == "auto":
            self.backend = _detect_backend(self.model_name)
        else:
            self.backend = ModelBackend(backend_str)

        quant_str = quantization or os.getenv("VLM_QUANTIZATION", "none")
        self.quantization = QuantizationMode(quant_str)

        if compile_model is not None:
            self.compile_model = compile_model
        else:
            self.compile_model = os.getenv("VLM_COMPILE", "false").lower() == "true"

        logger.info(
            f"VLMModel initialized: model={self.model_name}, "
            f"device={self.device}, backend={self.backend.value}, "
            f"quantization={self.quantization.value}, compile={self.compile_model}"
        )

        if not lazy_load:
            self._load_model()

    def _load_model(self) -> None:
        """
        Load the model and processor from HuggingFace.

        Dispatches to the appropriate loader based on self.backend,
        then optionally applies quantization and torch.compile().
        """
        if self._loaded:
            return

        logger.info(
            f"Loading VLM: {self.model_name} on {self.device} "
            f"(backend={self.backend.value}, "
            f"quantization={self.quantization.value})..."
        )

        try:
            import torch  # noqa: F401
            from transformers import AutoProcessor
        except ImportError as e:
            raise ImportError(
                "VLM dependencies not installed. Run: "
                "uv sync --extra vlm"
            ) from e

        torch_dtype = self._resolve_dtype()
        quantization_config = self._build_quantization_config()

        if self.backend == ModelBackend.QWEN:
            self._load_qwen(torch_dtype, quantization_config)
        else:
            self._load_transformers(torch_dtype, quantization_config)

        # Apply torch.compile() for graph-level optimization
        if self.compile_model:
            self._apply_compile()

        self.processor = AutoProcessor.from_pretrained(
            self.model_name, trust_remote_code=True
        )

        self._loaded = True
        logger.info(f"VLM loaded successfully on {self.device}")

    # ------------------------------------------------------------------
    # Backend-specific loaders
    # ------------------------------------------------------------------

    def _load_qwen(self, torch_dtype, quantization_config=None) -> None:
        """Load using Qwen-specific class for optimized performance."""
        from transformers import Qwen2VLForConditionalGeneration

        load_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": "auto" if self.device == "cuda" else None,
            "trust_remote_code": True,
        }
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
            # bitsandbytes requires device_map="auto"
            load_kwargs["device_map"] = "auto"

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name, **load_kwargs
        )
        if self.device != "cuda" and quantization_config is None:
            self.model = self.model.to(self.device)

    def _load_transformers(self, torch_dtype, quantization_config=None) -> None:
        """Load using generic AutoModel — works with most HuggingFace VLMs."""
        from transformers import AutoModelForVision2Seq

        load_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": "auto" if self.device == "cuda" else None,
            "trust_remote_code": True,
        }
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
            load_kwargs["device_map"] = "auto"

        try:
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name, **load_kwargs
            )
        except (ValueError, KeyError):
            from transformers import AutoModelForCausalLM

            logger.info(
                f"AutoModelForVision2Seq failed for {self.model_name}, "
                "falling back to AutoModelForCausalLM"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, **load_kwargs
            )

        if self.device != "cuda" and quantization_config is None:
            self.model = self.model.to(self.device)

    # ------------------------------------------------------------------
    # Dtype helper
    # ------------------------------------------------------------------

    def _resolve_dtype(self):
        """Pick the right torch dtype for the target device."""
        import torch

        if self.device == "cuda":
            return torch.bfloat16
        elif self.device == "mps":
            return torch.float16
        return torch.float32

    def _build_quantization_config(self):
        """
        Build a BitsAndBytesConfig for INT8/INT4 quantization.

        Returns None when quantization is disabled or unavailable.
        Quantization requires CUDA — silently falls back on other devices.
        """
        if self.quantization == QuantizationMode.NONE:
            return None

        if self.device != "cuda":
            logger.warning(
                f"Quantization ({self.quantization.value}) requires CUDA, "
                f"but device is {self.device}. Skipping quantization."
            )
            return None

        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            logger.warning(
                "bitsandbytes not installed — quantization disabled. "
                "Install with: pip install bitsandbytes"
            )
            return None

        if self.quantization == QuantizationMode.INT8:
            logger.info("Using INT8 quantization (load_in_8bit)")
            return BitsAndBytesConfig(load_in_8bit=True)
        elif self.quantization == QuantizationMode.INT4:
            logger.info("Using INT4 quantization (load_in_4bit, nf4)")
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self._resolve_dtype(),
            )
        return None

    def _apply_compile(self) -> None:
        """
        Apply torch.compile() for graph-level optimization.

        Uses 'reduce-overhead' mode which is best for inference
        (CUDA graphs, kernel fusion). Falls back gracefully on
        older PyTorch or unsupported platforms.
        """
        try:
            import torch

            if not hasattr(torch, "compile"):
                logger.warning("torch.compile requires PyTorch 2.0+, skipping")
                return

            logger.info("Applying torch.compile(mode='reduce-overhead')...")
            self.model = torch.compile(self.model, mode="reduce-overhead")
            logger.info("torch.compile applied successfully")
        except Exception as e:
            logger.warning(f"torch.compile failed (non-fatal): {e}")

    @staticmethod
    def cleanup_gpu_memory() -> None:
        """
        Release unused GPU memory back to the OS.

        Call after inference on memory-constrained edge devices
        (Jetson Nano/Orin) to prevent OOM on subsequent jobs.
        """
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass

    # ------------------------------------------------------------------
    # Image decoding
    # ------------------------------------------------------------------

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
            if img.mode != "RGB":
                img = img.convert("RGB")
            return img
        except Exception as e:
            logger.error(f"Error decoding image: {e}")
            return None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        image_base64: str,
        prompt: str,
        max_new_tokens: int = 256,
        detection_context: Optional[dict] = None,
    ) -> dict:
        """
        Generate text response given an image and prompt.

        Uses the standard OpenAI-style multimodal message format
        supported by most HuggingFace VLMs.

        Args:
            image_base64: Base64-encoded image.
            prompt: User's text prompt/question.
            max_new_tokens: Maximum tokens to generate.
            detection_context: Optional YOLO detection results to include
                              in the prompt for grounded reasoning.

        Returns:
            dict with 'response' or 'error' key.
        """
        if not self._loaded:
            self._load_model()

        image = self.decode_image(image_base64)
        if image is None:
            return {"error": "Invalid image data"}

        system_prompt = self._build_prompt(prompt, detection_context)

        try:
            import torch

            # Standard multimodal message format (works across models)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": system_prompt},
                    ],
                }
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = self.processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
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

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def is_loaded(self) -> bool:
        """Check if the model is currently loaded."""
        return self._loaded

    def get_info(self) -> dict:
        """Get model information for health checks."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "backend": self.backend.value,
            "quantization": self.quantization.value,
            "compiled": self.compile_model,
            "loaded": self._loaded,
        }
