#!/bin/bash
set -e

echo "=== Karna OCR — Starting Document Intelligence Engine ==="

# Start vLLM in background
echo "[1/2] Starting vLLM model server on port ${VLLM_PORT:-8000}..."
python -m vllm.entrypoints.openai.api_server \
    --model /app/model \
    --served-model-name karna-ocr \
    --port ${VLLM_PORT:-8000} \
    --tensor-parallel-size ${TENSOR_PARALLEL:-1} \
    --max-model-len ${MAX_MODEL_LEN:-4096} \
    --gpu-memory-utilization ${GPU_MEM_UTIL:-0.92} \
    --trust-remote-code \
    --dtype bfloat16 \
    &

VLLM_PID=$!

# Wait for vLLM to be ready
echo "Waiting for model to load..."
for i in $(seq 1 120); do
    if curl -sf http://localhost:${VLLM_PORT:-8000}/health > /dev/null 2>&1; then
        echo "vLLM ready after ${i}s"
        break
    fi
    sleep 1
done

# Start FastAPI
echo "[2/2] Starting Karna OCR API on port ${API_PORT:-8080}..."
python -m uvicorn api:app --host 0.0.0.0 --port ${API_PORT:-8080} &
API_PID=$!

echo "=== Karna OCR Ready ==="
echo "  Model API:    http://localhost:${VLLM_PORT:-8000}"
echo "  Document API: http://localhost:${API_PORT:-8080}"
echo "  Docs:         http://localhost:${API_PORT:-8080}/docs"

wait $VLLM_PID $API_PID
