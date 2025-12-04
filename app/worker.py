import json
import logging
import os
import signal
import sys
import time
import traceback

import redis

from app.vision import VisionModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration from environment variables
redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
queue_name = os.getenv("QUEUE_NAME", "vlm_queue")
redis_password = os.getenv("REDIS_PASSWORD", None)

# Global variables (will be initialized in main())
redis_client = None
vision_engine = None
shutdown_requested = False

def parse_redis_url(url: str):
    """Parse Redis URL into host, port, and db"""
    if url.startswith("redis://"):
        parts = url.replace("redis://", "").split("/")
        host_port = parts[0].split(":")
        redis_host = host_port[0] if len(host_port) > 0 else "redis"
        redis_port = int(host_port[1]) if len(host_port) > 1 else 6379
        redis_db = int(parts[1]) if len(parts) > 1 else 0
    else:
        redis_host = "redis"
        redis_port = 6379
        redis_db = 0
    return redis_host, redis_port, redis_db

def connect_to_redis():
    """Connect to Redis with retry logic"""
    global redis_client
    
    redis_host, redis_port, redis_db = parse_redis_url(redis_url)
    max_retries = 5
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                password=redis_password,
                decode_responses=False,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True
            )
            redis_client.ping()
            logger.info(f"Connected to Redis at {redis_host}:{redis_port}")
            return True
        except redis.exceptions.ConnectionError as e:
            if attempt < max_retries - 1:
                logger.warning(f"Redis connection failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error(f"Failed to connect to Redis after {max_retries} attempts: {e}")
                return False
        except Exception as e:
            logger.error(f"An unexpected error occurred during Redis connection: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                return False
    
    return False

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global shutdown_requested
    logger.info(f"Shutdown signal ({signum}) received. Finishing current job...")
    shutdown_requested = True

def main():
    """Main worker loop - only runs when script is executed directly"""
    global redis_client, vision_engine, shutdown_requested
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Connect to Redis
    if not connect_to_redis():
        logger.critical("Worker could not connect to Redis. Exiting.")
        sys.exit(1)
    
    # Initialize vision model
    logger.info("Worker initializing...")
    vision_engine = VisionModel()
    logger.info("Worker ready for jobs!")
    
    retry_delay = 2
    
    while not shutdown_requested:
        try:
            result = redis_client.brpop(queue_name, timeout=1)
            
            if result is None:
                continue
            
            _, job_json = result
            job = json.loads(job_json)
            job_id = job.get('id', 'unknown')
            logger.info(f"Processing job: {job_id}")
            start_time = time.time()
            
            try:
                conf_threshold = job.get("confidence_threshold", 0.5)
                result_data = vision_engine.predict(
                    job['image'],
                    conf_threshold=conf_threshold,
                )

                latency = round(time.time() - start_time, 4)
                result_data["latency_seconds"] = latency

                output = {
                    "status": "success",
                    "vision_result": result_data,
                    "vlm_result": "VLM not connected yet",
                }

                redis_client.setex(f"result:{job_id}", 3600, json.dumps(output))
                logger.info(f"Job {job_id} finished in {latency}s")

            except Exception as e:
                logger.error(f"Job {job_id} failed: {e}")
                traceback.print_exc()
                error_output = {"status": "failed", "error": str(e)}
                redis_client.setex(f"result:{job_id}", 3600, json.dumps(error_output))
        
        except redis.exceptions.ConnectionError as e:
            logger.error(f"Redis connection lost in worker loop: {e}. Attempting to reconnect...")
            if not connect_to_redis():
                logger.error("Failed to reconnect to Redis. Exiting.")
                break
        except Exception as e:
            logger.error(f"An unexpected error occurred in worker loop: {e}")
            traceback.print_exc()
            time.sleep(1)

    logger.info("Worker shutting down gracefully...")

if __name__ == "__main__":
    main()
