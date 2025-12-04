import cv2
import numpy as np
import base64
from ultralytics import YOLO

class VisionModel:
    def __init__(self, model_name='yolov8n.pt'):
        print(f"Loading Vision Model: {model_name}...")
        self.model = YOLO(model_name)
        print("Vision Model loaded successfully!")

    def decode_image(self, base64_string: str):
        """Convert Base64 string to OpenCV image"""
        try:
            img_data = base64.b64decode(base64_string)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            print(f"Error decoding image: {e}")
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