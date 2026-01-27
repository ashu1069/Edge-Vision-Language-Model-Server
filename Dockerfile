# Dockerfile
FROM python:3.10-slim

# 1. Install system dependencies
# FIX: 'libgl1-mesa-glx' is removed. We use 'libgl1' and 'libglib2.0-0' instead.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Install uv (The blazing fast package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# 3. Set up the application directory
WORKDIR /app

# 4. Copy ONLY the project definition first (Optimization for Caching)
COPY pyproject.toml .

# 5. Install dependencies using uv (including VLM extras)
# --system installs into the container's global python environment
RUN uv pip install --system ".[vlm]"

# 6. Copy the rest of the code
COPY . .

# 7. Start the app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]