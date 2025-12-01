import redis
import time
import json

from app.vision import VisionModel

# Connect to the same Redis instance
redis_client = redis.Redis(host='redis', port=6379, db=0)
print("Worker initializing...")
vision_engine = VisionModel()
print("Worker ready for jobs!")

while True:
    # 1. Block and wait for a job from the 'right' of the list
    # 'brpop' removes an item from the queue. If empty, it waits.
    _, job_json = redis_client.brpop("vlm_queue")
    
    job = json.loads(job_json)
    print(f"Processing job: {job['id']}")
    start_time = time.time()
    
    # RUN INFERENCE
    try:
        result_data = vision_engine.predict(
            job['image'],
            conf_threshold=0.5
        )

        # Add latency metrics
        latency = round(time.time() - start_time, 4)
        result_data["latency_seconds"] = latency

        output = {
            "status": "success",
            "vision_result": result_data,
            # Placeholder for Phase 3
            "vlm_result": "VLM not connected yet"
        }

    except Exception as e:
        print(f"Job failed: {e}")
        output = {"status": "failed", "error": str(e)}

    
    # Store result where API can find it (expire in 1 hour)
    redis_client.setex(f"result:{job['id']}", 3600, json.dumps(output))
    print(f"Job {job['id']} finished in {latency}s")