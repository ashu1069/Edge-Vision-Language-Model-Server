"""
YOLO-based object detection model wrapper.

Provides object detection using YOLOv8 with support for multiple
inference backends: PyTorch (.pt), TensorRT (.engine), and ONNX (.onnx).

On NVIDIA Jetson / edge GPUs, exporting to TensorRT yields 3-5x
speedup over PyTorch inference.
"""

import base64
import logging
import os
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# Supported export formats and their file extensions
_FORMAT_EXT = {
    "engine": ".engine",
    "tensorrt": ".engine",
    "onnx": ".onnx",
    "torchscript": ".torchscript",
    "openvino": "_openvino_model",
}


class VisionModel:
    """
    YOLOv8 object detection model wrapper with TensorRT / ONNX support.

    Loads the best available model format automatically:
      1. If a .engine file exists next to the .pt file → use TensorRT
      2. If YOLO_EXPORT_FORMAT is set → export on first load, then use engine
      3. Otherwise → standard PyTorch inference

    Environment variables:
        YOLO_MODEL:         Model path or name (default: 'yolov8n.pt')
        YOLO_EXPORT_FORMAT: Auto-export format on startup ('engine', 'onnx', etc.)
        YOLO_HALF:          Use FP16 inference when available ('true'/'false')
        YOLO_IMGSZ:         Input image size for export (default: 640)
        YOLO_WARMUP:        Number of warmup runs after loading (default: 1)
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        export_format: Optional[str] = None,
        half: Optional[bool] = None,
        imgsz: Optional[int] = None,
    ):
        self.model_name = model_name or os.getenv("YOLO_MODEL", "yolov8n.pt")
        self.export_format = export_format or os.getenv("YOLO_EXPORT_FORMAT", None)
        self.half = half if half is not None else os.getenv("YOLO_HALF", "false").lower() == "true"
        self.imgsz = imgsz or int(os.getenv("YOLO_IMGSZ", "640"))
        self.warmup_runs = int(os.getenv("YOLO_WARMUP", "1"))

        model_path = self._resolve_model_path()
        logger.info(f"Loading Vision Model: {model_path}...")
        self.model = YOLO(model_path)

        self._warmup()
        logger.info("Vision Model loaded successfully!")

    # ------------------------------------------------------------------
    # Model resolution & export
    # ------------------------------------------------------------------

    def _resolve_model_path(self) -> str:
        """
        Determine the best model file to load.

        Priority:
          1. If self.model_name already points to an engine/onnx file → use it
          2. If a pre-exported engine exists alongside the .pt → use it
          3. If YOLO_EXPORT_FORMAT is set → export now and return engine path
          4. Fall back to the original .pt file
        """
        path = Path(self.model_name)

        # Already an optimized format
        if path.suffix in (".engine", ".onnx", ".torchscript"):
            return str(path)

        # Check for pre-exported engine next to the .pt
        if self.export_format:
            ext = _FORMAT_EXT.get(self.export_format, f".{self.export_format}")
            engine_path = path.with_suffix(ext)
            if engine_path.exists():
                logger.info(f"Found pre-exported model: {engine_path}")
                return str(engine_path)

            # Export on first run
            return self._export_model(str(path), self.export_format)

        # Default: plain PyTorch
        return str(path)

    def _export_model(self, pt_path: str, fmt: str) -> str:
        """
        Export a .pt model to the requested format via ultralytics.

        Returns the path to the exported model file.
        """
        logger.info(f"Exporting {pt_path} → {fmt} (imgsz={self.imgsz}, half={self.half})...")
        t0 = time.time()

        tmp_model = YOLO(pt_path)
        export_path = tmp_model.export(
            format=fmt,
            imgsz=self.imgsz,
            half=self.half,
        )

        elapsed = round(time.time() - t0, 1)
        logger.info(f"Export complete in {elapsed}s → {export_path}")
        return str(export_path)

    def _warmup(self) -> None:
        """Run dummy inference(s) to warm up the engine / JIT compilation."""
        if self.warmup_runs <= 0:
            return
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        for i in range(self.warmup_runs):
            self.model.predict(dummy, verbose=False)
        logger.info(f"Warmup complete ({self.warmup_runs} run(s))")

    # ------------------------------------------------------------------
    # Image decoding
    # ------------------------------------------------------------------

    def decode_image(self, base64_string: str) -> Optional[np.ndarray]:
        """Convert Base64 string to OpenCV image."""
        try:
            img_data = base64.b64decode(base64_string)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            logger.error(f"Error decoding image: {e}")
            return None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, image_base64: str, conf_threshold: float = 0.5) -> dict:
        """
        Run object detection on base64 encoded image.

        Uses FP16 inference when self.half is True and the backend
        supports it (TensorRT, CUDA).
        """
        img = self.decode_image(image_base64)
        if img is None:
            return {"error": "Invalid image data"}

        results = self.model.predict(
            img,
            conf=conf_threshold,
            half=self.half,
            verbose=False,
        )
        detections = []
        result = results[0]

        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            confidence = float(box.conf[0])

            detections.append({
                "class": class_name,
                "confidence": round(confidence, 2),
                "box": box.xywhn[0].tolist(),
            })

        return {"detections": detections, "count": len(detections)}

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def get_info(self) -> dict:
        """Model metadata for health checks and debugging."""
        model_path = str(self.model.ckpt_path) if hasattr(self.model, "ckpt_path") else self.model_name
        return {
            "model": model_path,
            "export_format": self.export_format,
            "half": self.half,
            "imgsz": self.imgsz,
        }
