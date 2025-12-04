#!/usr/bin/env python3
"""
Annotate inference results: Draw bounding boxes on image and save JSON
Usage: python3 annotate_result.py <image_file> <request_id>

Requirements:
  uv pip install opencv-python requests
  OR
  uv pip install pillow requests (for PIL-based version)
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import requests

try:
    import cv2
    import numpy as np
    USE_OPENCV = True
except ImportError:
    try:
        from PIL import Image, ImageDraw, ImageFont
        USE_OPENCV = False
        print("Note: OpenCV not found, using PIL instead. Install opencv-python for better performance.")
    except ImportError:
        print("Error: Neither opencv-python nor Pillow is installed.")
        print("Install one of them:")
        print("  uv pip install opencv-python")
        print("  OR")
        print("  uv pip install pillow")
        sys.exit(1)

def get_result(request_id: str):
    """Fetch result from API"""
    response = requests.get(f"http://localhost:8000/result/{request_id}")
    if response.status_code != 200:
        print(f"Error: Failed to fetch result: {response.text}")
        sys.exit(1)
    
    data = response.json()
    if data["status"] != "completed":
        print(f"Error: Request not completed. Status: {data['status']}")
        sys.exit(1)
    
    return data["data"]

def denormalize_box(box, img_width, img_height):
    """
    Convert normalized xywhn coordinates to pixel coordinates
    box format: [center_x, center_y, width, height] (all normalized 0-1)
    Returns: [x1, y1, x2, y2] in pixels
    """
    center_x, center_y, width, height = box
    
    # Convert to pixel coordinates
    center_x_px = center_x * img_width
    center_y_px = center_y * img_height
    width_px = width * img_width
    height_px = height * img_height
    
    # Calculate top-left and bottom-right corners
    x1 = int(center_x_px - width_px / 2)
    y1 = int(center_y_px - height_px / 2)
    x2 = int(center_x_px + width_px / 2)
    y2 = int(center_y_px + height_px / 2)
    
    return x1, y1, x2, y2

def get_class_color(class_name: str):
    """Get color for a class"""
    colors = {
        "person": (0, 255, 0),      # Green
        "car": (255, 0, 0),         # Blue
        "bicycle": (0, 0, 255),     # Red
        "dog": (255, 165, 0),       # Orange
        "cat": (255, 192, 203),     # Pink
    }
    return colors.get(class_name.lower(), (255, 255, 0))  # Yellow default

def draw_bounding_boxes_opencv(image_path: str, detections: list, output_path: str):
    """Draw bounding boxes using OpenCV"""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        sys.exit(1)
    
    img_height, img_width = img.shape[:2]
    
    # Draw each detection
    for detection in detections:
        class_name = detection["class"]
        confidence = detection["confidence"]
        box = detection["box"]
        
        # Convert normalized coordinates to pixel coordinates
        x1, y1, x2, y2 = denormalize_box(box, img_width, img_height)
        
        # Get color (BGR format for OpenCV)
        color_bgr = get_class_color(class_name)
        color = (color_bgr[2], color_bgr[1], color_bgr[0])  # RGB to BGR
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label text
        label = f"{class_name} {confidence:.2f}"
        
        # Calculate text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        # Draw label background
        cv2.rectangle(
            img,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            img,
            label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),  # Black text
            2
        )
    
    # Save annotated image
    cv2.imwrite(output_path, img)
    print(f"✓ Annotated image saved: {output_path}")

def draw_bounding_boxes_pil(image_path: str, detections: list, output_path: str):
    """Draw bounding boxes using PIL"""
    # Read image
    img = Image.open(image_path)
    img_width, img_height = img.size
    draw = ImageDraw.Draw(img)
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font = ImageFont.load_default()
    
    # Draw each detection
    for detection in detections:
        class_name = detection["class"]
        confidence = detection["confidence"]
        box = detection["box"]
        
        # Convert normalized coordinates to pixel coordinates
        x1, y1, x2, y2 = denormalize_box(box, img_width, img_height)
        
        # Get color (RGB format for PIL)
        color = get_class_color(class_name)
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Prepare label text
        label = f"{class_name} {confidence:.2f}"
        
        # Get text size
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Draw label background
        draw.rectangle(
            [x1, y1 - text_height - 5, x1 + text_width + 4, y1],
            fill=color
        )
        
        # Draw label text
        draw.text(
            (x1 + 2, y1 - text_height - 3),
            label,
            fill=(0, 0, 0),  # Black text
            font=font
        )
    
    # Save annotated image
    img.save(output_path)
    print(f"✓ Annotated image saved: {output_path}")

def draw_bounding_boxes(image_path: str, detections: list, output_path: str):
    """Draw bounding boxes using available library"""
    if USE_OPENCV:
        draw_bounding_boxes_opencv(image_path, detections, output_path)
    else:
        draw_bounding_boxes_pil(image_path, detections, output_path)

def save_json(data: dict, output_path: str):
    """Save JSON data to file"""
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✓ JSON data saved: {output_path}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 annotate_result.py <image_file> <request_id>")
        print("Example: python3 annotate_result.py test.jpg abc123-def456-...")
        sys.exit(1)
    
    image_path = sys.argv[1]
    request_id = sys.argv[2]
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)
    
    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Get result from API
    print(f"Fetching result for request ID: {request_id}")
    result_data = get_result(request_id)
    
    if result_data["status"] != "success":
        print(f"Error: Inference failed: {result_data.get('error', 'Unknown error')}")
        sys.exit(1)
    
    vision_result = result_data["vision_result"]
    detections = vision_result.get("detections", [])
    
    if not detections:
        print("Warning: No detections found in the image")
    
    # Generate output filenames
    image_name = Path(image_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    annotated_image_path = results_dir / f"{image_name}_annotated_{timestamp}.jpg"
    json_output_path = results_dir / f"{image_name}_result_{timestamp}.json"
    
    # Draw bounding boxes
    if detections:
        draw_bounding_boxes(image_path, detections, str(annotated_image_path))
    else:
        print("No detections to annotate")
    
    # Save JSON
    output_data = {
        "request_id": request_id,
        "timestamp": datetime.now().isoformat(),
        "image_file": image_path,
        "detections": detections,
        "count": vision_result.get("count", 0),
        "latency_seconds": vision_result.get("latency_seconds", 0),
        "confidence_threshold": vision_result.get("confidence_threshold", 0.5)
    }
    
    save_json(output_data, str(json_output_path))
    
    print("\n" + "="*60)
    print("Results Summary")
    print("="*60)
    print(f"Detections: {len(detections)}")
    for i, det in enumerate(detections, 1):
        print(f"  {i}. {det['class']} (confidence: {det['confidence']:.2f})")
    print(f"\nFiles saved to: {results_dir}/")
    print(f"  - Annotated image: {annotated_image_path.name}")
    print(f"  - JSON data: {json_output_path.name}")

if __name__ == "__main__":
    main()

