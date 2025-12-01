# Edge-Vision-Language-Model-Server

## Phase 1: The Skeleton (Infrastructure & API)
A working Dockerized FastAPI server that accepts requests, puts them in a Redis Queue, and returns 'fake' result for now (no AI models yet).

## Phase 2: The Eyes (Vision Worker)
Replace the facke vision worker with a TensorRT-optimized YOLOv8 container

## Phase 3: The Brain (VLM Worker)
Replace the face VLM worker with a vLLM server (e.g., LlaVA, PaliGemma)

## Phase 4: The Optimization (C++ Glue)
Write a custom C++ post-processor to bridge the two models efficienctly
