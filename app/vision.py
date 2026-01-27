"""
YOLO-based object detection model wrapper.

Provides object detection using YOLOv8 with base64 image input support.
"""

import base64
import logging

import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class VisionModel:
    """YOLOv8 object detection model wrapper."""

    def __init__(self, model_name: str = "yolov8n.pt"):
        logger.info(f"Loading Vision Model: {model_name}...")
        self.model = YOLO(model_name)
        logger.info("Vision Model loaded successfully!")

    def decode_image(self, base64_string: str):
        """Convert Base64 string to OpenCV image."""
        try:
            img_data = base64.b64decode(base64_string)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            logger.error(f"Error decoding image: {e}")
            return None

    def predict(self, image_base64: str, conf_threshold: float = 0.5):
        """Run object detection on base64 encoded image"""
        img = self.decode_image(image_base64)
        if img is None:
            return {"error": "Invalid image data"}

        results = self.model.predict(img, conf=conf_threshold, verbose=False)
        detections = []
        result = results[0]

        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            confidence = float(box.conf[0])

            detections.append({
                "class": class_name,
                "confidence": round(confidence, 2),
                "box": box.xywhn[0].tolist()
            })

        return {"detections": detections, "count": len(detections)}