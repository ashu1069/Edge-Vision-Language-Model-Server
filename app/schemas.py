from pydantic import BaseModel, Field
from typing import Optional
'''
We never guess what data looks like. We define it strictly. 
Here, we're defining what the user must send us.
'''
class InferenceRequest(BaseModel):
    image_base64: str # The image encoded as a string
    prompt: str # The user's question (e.g., Is the person wearing a helmet?)
    confidence_threshold: float = 0.5 

class InferenceResponse(BaseModel):
    request_id: str
    status: str # "queued", "processing", "completed"
    result: Optional[dict] = None