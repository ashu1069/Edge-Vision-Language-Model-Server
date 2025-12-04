from typing import Optional

from pydantic import BaseModel


class InferenceRequest(BaseModel):
    """Request schema for inference endpoint"""
    image_base64: str
    prompt: str
    confidence_threshold: float = 0.5


class InferenceResponse(BaseModel):
    """Response schema for inference endpoint"""
    request_id: str
    status: str
    result: Optional[dict] = None