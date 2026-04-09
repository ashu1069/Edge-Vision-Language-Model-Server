"""
Background worker for processing inference jobs.

Pulls jobs from Redis queue, routes them based on prompt analysis,
and executes the appropriate model(s) (YOLO, VLM, or both).
"""

import json
import logging
import os
import signal
import sys
import time
import traceback
from typing import Any, Optional

import redis

from app.redis_utils import parse_redis_url
from app.router import TaskType, route_prompt
from app.vision import VisionModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration from environment variables
redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
queue_name = os.getenv("QUEUE_NAME", "vlm_queue")
redis_password = os.getenv("REDIS_PASSWORD", None)
vlm_enabled = os.getenv("VLM_ENABLED", "true").lower() == "true"
lazy_load_vlm = os.getenv("LAZY_LOAD_VLM", "true").lower() == "true"

# Global variables (will be initialized in main())
redis_client: Optional[redis.Redis] = None
vision_engine: Optional[VisionModel] = None
vlm_engine: Optional[Any] = None  # Lazy import to avoid loading torch at startup
shutdown_requested = False



def connect_to_redis() -> bool:
    """
    Connect to Redis with retry logic.
    
    Returns:
        True if connection successful, False otherwise.
    """
    global redis_client
    
    host, port, db = parse_redis_url(redis_url)
    max_retries = 5
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            redis_client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=redis_password,
                decode_responses=False,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True
            )
            redis_client.ping()
            logger.info(f"Connected to Redis at {host}:{port}")
            return True
        except redis.exceptions.ConnectionError as e:
            if attempt < max_retries - 1:
                logger.warning(
                    f"Redis connection failed (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {retry_delay} seconds..."
                )
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


def get_vlm_engine():
    """
    Get or initialize the VLM engine (lazy loading).
    
    Returns:
        VLMModel instance, or None if VLM is disabled or unavailable.
    """
    global vlm_engine
    
    if not vlm_enabled:
        return None
    
    if vlm_engine is not None:
        return vlm_engine
    
    try:
        from app.vlm import VLMModel
        logger.info("Initializing VLM engine...")
        vlm_engine = VLMModel(lazy_load=lazy_load_vlm)
        if not lazy_load_vlm:
            # Force load the model now
            vlm_engine._load_model()
        logger.info("VLM engine initialized successfully")
        return vlm_engine
    except ImportError as e:
        logger.warning(f"VLM dependencies not available: {e}. VLM will be disabled.")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize VLM engine: {e}")
        return None


def _cleanup_gpu_memory() -> None:
    """Release unused GPU memory after each inference job."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_requested
    logger.info(f"Shutdown signal ({signum}) received. Finishing current job...")
    shutdown_requested = True


def process_job(job: dict) -> dict:
    """
    Process a single inference job.
    
    Routes the job based on prompt analysis and executes
    the appropriate model(s).
    
    Args:
        job: Job dictionary with 'id', 'image', 'prompt', 'confidence_threshold'
        
    Returns:
        Result dictionary with 'status', 'task_type', 'vision_result', 'vlm_result'
    """
    image_base64 = job.get("image", "")
    prompt = job.get("prompt", "")
    conf_threshold = job.get("confidence_threshold", 0.5)
    
    # Route the prompt to determine task type
    task_type = route_prompt(prompt)
    logger.info(f"Routed prompt to task type: {task_type.value}")
    
    # If VLM is disabled, fall back to detection only for VLM tasks
    vlm = get_vlm_engine()
    if vlm is None and task_type in (TaskType.VLM_ONLY, TaskType.DETECTION_AND_VLM):
        logger.warning("VLM not available, falling back to detection only")
        task_type = TaskType.DETECTION_ONLY
    
    vision_result = None
    vlm_result = None
    
    # Execute based on task type
    if task_type == TaskType.DETECTION_ONLY:
        vision_result = vision_engine.predict(
            image_base64,
            conf_threshold=conf_threshold
        )
        
    elif task_type == TaskType.VLM_ONLY:
        vlm_response = vlm.predict(
            image_base64,
            prompt=prompt
        )
        if "error" in vlm_response:
            vlm_result = f"Error: {vlm_response['error']}"
        else:
            vlm_result = vlm_response.get("response", "")
            
    elif task_type == TaskType.DETECTION_AND_VLM:
        # Run YOLO first
        vision_result = vision_engine.predict(
            image_base64,
            conf_threshold=conf_threshold
        )
        
        # Pass detection context to VLM
        vlm_response = vlm.predict(
            image_base64,
            prompt=prompt,
            detection_context=vision_result if "error" not in vision_result else None
        )
        if "error" in vlm_response:
            vlm_result = f"Error: {vlm_response['error']}"
        else:
            vlm_result = vlm_response.get("response", "")
    
    return {
        "task_type": task_type.value,
        "vision_result": vision_result,
        "vlm_result": vlm_result,
    }


def main():
    """Main worker loop - only runs when script is executed directly."""
    global redis_client, vision_engine, shutdown_requested
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Connect to Redis
    if not connect_to_redis():
        logger.critical("Worker could not connect to Redis. Exiting.")
        sys.exit(1)
    
    # Initialize vision model (always needed)
    logger.info("Worker initializing...")
    vision_engine = VisionModel()
    
    # Optionally pre-load VLM if not lazy loading
    if vlm_enabled and not lazy_load_vlm:
        get_vlm_engine()
    
    logger.info(f"Worker ready for jobs! (VLM enabled: {vlm_enabled})")
    
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
                # Process the job through router
                result_data = process_job(job)

                latency = round(time.time() - start_time, 4)

                output = {
                    "status": "success",
                    "task_type": result_data["task_type"],
                    "vision_result": result_data["vision_result"],
                    "vlm_result": result_data["vlm_result"],
                    "latency_seconds": latency,
                }

                redis_client.setex(f"result:{job_id}", 3600, json.dumps(output))
                logger.info(
                    f"Job {job_id} finished in {latency}s "
                    f"(task_type: {result_data['task_type']})"
                )

            except Exception as e:
                logger.error(f"Job {job_id} failed: {e}")
                traceback.print_exc()
                error_output = {"status": "failed", "error": str(e)}
                redis_client.setex(f"result:{job_id}", 3600, json.dumps(error_output))
            finally:
                # Free GPU memory after each job (critical on edge devices)
                _cleanup_gpu_memory()

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
