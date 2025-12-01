import cv2
import numpy as np
import base64
from ultralytics import YOLO

class VisionModel:
    def __init__(self, model_name = 'yolov8n.pt'):
        print(f"Loading Vision Model: {model_name}...")
        # This downloads the model weights on the first run
        self.model = YOLO(model_name)
        print("Vision Model loaded succesfully!")

    def decode_image(self, base64_string: str):
        '''
        Converts Base64 string back to an OpenCV image
        '''
        try:
            # Decode base64 string to bytes
            img_data = base64.b64decode(base64_string)
            # Convert bytes to numpy array
            nparr = np.frombuffer(img_data, np.uint8)
            # Decode image
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            print(f"Error decoding image: {e}")
            return None

    def predict(self, image_base64: str, conf_threshold: float = 0.5):
        # 1. Decode
        img = self.decode_image(image_base64)
        if img is None:
            return {"error": "Invalid image data"}

        # 2. Run inference
        # verbose = False keeps logs clean
        results = self.model.predict(img, conf_threshold=conf_threshold, verbose=False)

        # 3. Parse results (convert complex YOLO objects to simple JSON)
        detections = []
        result = results[0] # We've only sent one image

        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            confidence = float(box.conf[0])

            detections.append({
                "class": class_name,
                "confidence": round(confidence, 2),
                # Normalized coordinates are better for API responses than pixels
                "box": box.xywhn[0].tolist()
            })

        return {"detections": detections, "count": len(detections)}