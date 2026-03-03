# Karna OCR — Document Intelligence Engine

AI-powered document extraction that reads **anything** — forms, invoices, contracts, handwritten notes, tables, receipts, and more. Returns structured JSON with confidence scores.

Built on [Qwen3-VL-32B](https://huggingface.co/Qwen/Qwen3-VL-32B), a state-of-the-art vision-language model.

## Quick Start

### Requirements
- NVIDIA GPU with **48GB+ VRAM** (A6000, A100, etc.)
- Docker with NVIDIA Container Toolkit
- ~70GB disk space

### Run

```bash
git clone https://github.com/Viraj0518/Karna-OCR.git
cd Karna-OCR
docker compose up --build
```

First build downloads the model weights (~63GB) — grab a coffee. After that, starts in ~60s.

- **API Docs:** http://localhost:8080/docs
- **Health:** http://localhost:8080/health

## API

### `POST /extract/upload` — Upload a document

```bash
curl -X POST http://localhost:8080/extract/upload \
  -F "file=@my_form.png" \
  -F "prompt_template=form"
```

### `POST /extract` — Base64 or URL

```python
import requests, base64

with open("invoice.png", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

resp = requests.post("http://localhost:8080/extract", json={
    "image_base64": b64,
    "prompt_template": "invoice"
})

print(resp.json()["data"])
```

### `POST /extract/batch` — Multiple documents

```bash
curl -X POST http://localhost:8080/extract/batch \
  -F "files=@doc1.png" \
  -F "files=@doc2.png" \
  -F "prompt_template=general"
```

## Prompt Templates

| Template | Use Case |
|---|---|
| `general` | Any document (default) |
| `form` | Structured forms with labeled fields |
| `invoice` | Invoices, receipts, purchase orders |
| `table` | Documents with tabular data |
| `handwriting` | Handwritten notes, letters, prescriptions |
| `contract` | Legal documents, agreements, NDAs |

Use `prompt_template` parameter, or pass your own `prompt` for full control.

## Response Format

```json
{
  "status": "ok",
  "data": {
    "Personal Information": {
      "Last Name": {"value": "STEINER", "confidence": 0.95, "type": "handwritten"},
      "First Name": {"value": "MICHAEL", "confidence": 0.92, "type": "handwritten"},
      "Email": {"value": "msteiner@karna.com", "confidence": 0.88, "type": "handwritten"}
    }
  },
  "raw_output": "...",
  "processing_time_ms": 8500,
  "model": "karna-ocr"
}
```

## CLI Scripts

```bash
# Single document
python scripts/ocr_extract.py my_form.png --prompt-template form

# Batch processing
python scripts/batch_process.py ./documents/ --output ./results/ --prompt-template invoice

# Custom prompt
python scripts/ocr_extract.py scan.png --prompt "Extract only names and dates" --output result.json
```

## Configuration

Environment variables (set in `docker-compose.yml`):

| Variable | Default | Description |
|---|---|---|
| `TENSOR_PARALLEL` | `1` | Number of GPUs (set to 2 for dual-GPU) |
| `MAX_MODEL_LEN` | `4096` | Max context length |
| `GPU_MEM_UTIL` | `0.92` | GPU memory utilization (0.0-1.0) |
| `API_PORT` | `8080` | API server port |
| `VLLM_PORT` | `8000` | vLLM model server port |

### Dual-GPU Setup

Edit `docker-compose.yml`:
```yaml
environment:
  - TENSOR_PARALLEL=2
deploy:
  resources:
    reservations:
      devices:
        - count: 2
```

## Benchmarks

Tested on WTC Health Program enrollment form (handwritten fields):

| Engine | Accuracy | Time | Structured Output |
|---|---|---|---|
| **Karna OCR** | **92%** | **~10s** | **✅ JSON with confidence** |
| Tesseract 5.5 | 25% | 4s | ❌ Raw text |
| EasyOCR | 33% | 45s | ❌ Raw text |
| Tesseract + TrOCR | 42% | 348s | ❌ Raw text |

## Architecture

```
Request → FastAPI (port 8080) → vLLM + Qwen3-VL-32B (port 8000) → Structured JSON
```

No preprocessing, no OCR pipeline, no post-processing regex. The vision-language model reads the document directly and returns structured data.

## License

Apache 2.0
