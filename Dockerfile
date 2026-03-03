FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/hf_cache
ENV MODEL_ID=Qwen/Qwen3-VL-32B
ENV VLLM_PORT=8000
ENV API_PORT=8080

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev git curl wget \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download model weights at build time (~63GB)
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('${MODEL_ID}', local_dir='/app/model', local_dir_use_symlinks=False)"

# Copy application code
COPY api.py .
COPY prompts/ prompts/
COPY scripts/ scripts/
COPY examples/ examples/
COPY start.sh .
RUN chmod +x start.sh

EXPOSE ${VLLM_PORT} ${API_PORT}

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:${API_PORT}/health || exit 1

CMD ["./start.sh"]
