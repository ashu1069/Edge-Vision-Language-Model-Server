"""
Pydantic schemas for API request/response validation.

Defines the structure of inference requests and responses,
including support for both YOLO detection and VLM reasoning results.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    """
    Request schema for inference endpoint.
    
    Attributes:
        image_base64: Base64-encoded image data (JPEG, PNG, etc.)
        prompt: Text prompt describing the task. The prompt is analyzed
                to determine whether to use YOLO, VLM, or both.
        confidence_threshold: Minimum confidence for YOLO detections.
                              Only applies when YOLO is used.
    """
    image_base64: str = Field(
        ...,
        description="Base64-encoded image data"
    )
    prompt: str = Field(
        ...,
        description="Text prompt describing the inference task"
    )
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for detections (0.0-1.0)"
    )


class InferenceResponse(BaseModel):
    """
    Response schema for inference submission.
    
    Returned immediately when a job is queued.
    
    Attributes:
        request_id: UUID for tracking the request
        status: Job status ('queued')
    """
    request_id: str = Field(
        ...,
        description="Unique identifier for tracking this request"
    )
    status: str = Field(
        ...,
        description="Current status of the request"
    )


class Detection(BaseModel):
    """
    Single object detection result.

    Attributes:
        class_name: Detected object class (e.g., 'person', 'car')
        confidence: Detection confidence score (0.0-1.0)
        box: Bounding box in normalized xywhn format
             [center_x, center_y, width, height]
    """

    model_config = {"populate_by_name": True}

    class_name: str = Field(
        ...,
        alias="class",
        description="Detected object class name",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Detection confidence score",
    )
    box: List[float] = Field(
        ...,
        description="Bounding box [center_x, center_y, width, height] normalized",
    )


class VisionResult(BaseModel):
    """
    YOLO detection results.
    
    Attributes:
        detections: List of detected objects
        count: Total number of detections
    """
    detections: List[Detection] = Field(
        default_factory=list,
        description="List of detected objects"
    )
    count: int = Field(
        default=0,
        ge=0,
        description="Total number of detections"
    )


class InferenceResult(BaseModel):
    """
    Complete inference result.
    
    Returned when polling for job results.
    
    Attributes:
        status: Completion status ('success', 'failed')
        task_type: Type of task executed ('detection', 'vlm', 'both')
        vision_result: YOLO detection results (if applicable)
        vlm_result: VLM text response (if applicable)
        latency_seconds: Total inference time
        error: Error message (if status is 'failed')
    """
    status: Literal["success", "failed"] = Field(
        ...,
        description="Completion status"
    )
    task_type: Optional[Literal["detection", "vlm", "both"]] = Field(
        default=None,
        description="Type of inference task executed"
    )
    vision_result: Optional[dict] = Field(
        default=None,
        description="YOLO detection results"
    )
    vlm_result: Optional[str] = Field(
        default=None,
        description="VLM text response"
    )
    latency_seconds: Optional[float] = Field(
        default=None,
        ge=0,
        description="Total inference time in seconds"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if status is 'failed'"
    )


class ResultResponse(BaseModel):
    """
    Response schema for result polling endpoint.
    
    Attributes:
        status: Job status ('processing', 'completed')
        data: Inference result data (when completed)
    """
    status: Literal["processing", "completed"] = Field(
        ...,
        description="Job processing status"
    )
    data: Optional[InferenceResult] = Field(
        default=None,
        description="Inference result data (present when status is 'completed')"
    )


class HealthResponse(BaseModel):
    """
    Response schema for health check endpoint.
    
    Attributes:
        status: Overall health status
        redis: Redis connection status
        vlm_enabled: Whether VLM is enabled
        vlm_loaded: Whether VLM model is loaded (if enabled)
    """
    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        ...,
        description="Overall service health status"
    )
    redis: Literal["healthy", "unhealthy", "unavailable"] = Field(
        ...,
        description="Redis connection status"
    )
    vlm_enabled: Optional[bool] = Field(
        default=None,
        description="Whether VLM is enabled"
    )
    vlm_loaded: Optional[bool] = Field(
        default=None,
        description="Whether VLM model is currently loaded"
    )
