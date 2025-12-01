from fastapi import FastAPI
from app.schemas import InferenceRequest, InferenceResponse
import uuid
import redis
import json

app = FastAPI(title="Edge VLM Infra")

# Connect to Redis (The "Waiting Room")
# host = 'redis' works because Docker networking resolves service names
redis_client = redis.Redis(host='redis', port=6379, db=0)

@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest):
    # 1. Generate a unique Ticket Number (UUID)
    request_id = str(uuid.uuid4())

    # 2. Package the job
    job_data ={
        "id": request_id,
        "image": request.image_base64,
        "prompt": request.prompt
    }

    # 3. Push to Redis Queue (push to the 'left' of the list)
    redis_client.lpush("vlm_queue", json.dumps(job_data))

    # 4. Immediately tell the user "We got it!" (Non-blocking)
    return InferenceResponse(
        request_id=request_id,
        status="queued"
    )

@app.get("/result/{request_id}")
async def get_result(request_id: str):
    # Check if result exists in Redis (Simulating a database here for simplicity)
    result = redis_client.get(f"result: {request_id}")

    if result:
        return {"status": "completed", "data": json.loads(result)}
    else:
        return {"status": "processing"}