import redis
import time
import json

# Connect to the same Redis instance
redis_client = redis.Redis(host='redis', port=6379, db=0)

print("👷 Worker started. Waiting for jobs...")

while True:
    # 1. Block and wait for a job from the 'right' of the list
    # 'brpop' removes an item from the queue. If empty, it waits.
    _, job_json = redis_client.brpop("vlm_queue")
    
    job = json.loads(job_json)
    print(f"Processing job: {job['id']}")
    
    # 2. Simulate heavy AI workload (Phase 2 & 3 will replace this)
    time.sleep(2) 
    
    # 3. Save fake result
    fake_output = {
        "detected_objects": ["person", "helmet"],
        "answer": "Yes, the person is wearing a helmet."
    }
    
    # 4. Store result where API can find it (expire in 1 hour)
    redis_client.setex(f"result:{job['id']}", 3600, json.dumps(fake_output))
    print(f"Job {job['id']} completed.")