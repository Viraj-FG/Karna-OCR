"""
Karna OCR — Document Intelligence API
Powered by Qwen3-VL-32B vision language model.

Extracts structured data from any document: forms, invoices, contracts,
handwritten notes, receipts, tables, medical records, and more.
"""

import base64, io, os, time, json, logging
from pathlib import Path
from typing import Optional

import requests
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("karna-ocr")

# ── Config ──────────────────────────────────────────────────────────────────

VLLM_URL = f"http://localhost:{os.getenv('VLLM_PORT', '8000')}/v1/chat/completions"
MODEL_NAME = "karna-ocr"
PROMPTS_DIR = Path(__file__).parent / "prompts"

# ── Prompt Library ──────────────────────────────────────────────────────────

def load_prompts():
    prompts = {}
    for f in PROMPTS_DIR.glob("*.txt"):
        prompts[f.stem] = f.read_text(encoding="utf-8").strip()
    return prompts

PROMPTS = load_prompts()

DEFAULT_PROMPT = PROMPTS.get("general", (
    "You are an expert document analysis AI. Extract ALL information from this document. "
    "For each field or data point, return a JSON object with:\n"
    '- "value": the extracted text\n'
    '- "confidence": a score from 0.0 to 1.0 (1.0 = perfectly clear, 0.5 = partially legible)\n'
    '- "type": "printed", "handwritten", "checkbox", or "stamp"\n'
    "Group fields logically. Include every piece of information visible in the document."
))

# ── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Karna OCR",
    description="Document Intelligence API — extract structured data from any document using AI vision.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Models ──────────────────────────────────────────────────────────────────

class ExtractionRequest(BaseModel):
    image_base64: Optional[str] = None
    image_url: Optional[str] = None
    prompt: Optional[str] = None
    prompt_template: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.1

class ExtractionResponse(BaseModel):
    status: str
    data: dict | str | None
    raw_output: str
    processing_time_ms: int
    prompt_used: str
    model: str = MODEL_NAME

# ── Helpers ─────────────────────────────────────────────────────────────────

def image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def call_vlm(b64: str, prompt: str, max_tokens: int = 4096, temperature: float = 0.1) -> str:
    resp = requests.post(VLLM_URL, json={
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": prompt}
        ]}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }, timeout=180)

    if resp.status_code != 200:
        raise HTTPException(502, f"Model server error: {resp.status_code} {resp.text[:500]}")

    return resp.json()["choices"][0]["message"]["content"]

def parse_json_output(text: str):
    """Try to extract JSON from model output, handling markdown fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = lines[1:]  # remove opening fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None

def resolve_prompt(custom: Optional[str], template: Optional[str]) -> str:
    if custom:
        return custom
    if template and template in PROMPTS:
        return PROMPTS[template]
    return DEFAULT_PROMPT

# ── Routes ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    try:
        r = requests.get(f"http://localhost:{os.getenv('VLLM_PORT', '8000')}/health", timeout=5)
        model_ok = r.status_code == 200
    except:
        model_ok = False
    return {
        "status": "ok" if model_ok else "degraded",
        "model_server": "up" if model_ok else "down",
        "model": MODEL_NAME,
        "prompts_available": list(PROMPTS.keys()),
    }

@app.get("/prompts")
def list_prompts():
    """List available prompt templates."""
    return {name: text[:200] + "..." if len(text) > 200 else text for name, text in PROMPTS.items()}

@app.post("/extract", response_model=ExtractionResponse)
async def extract_from_json(req: ExtractionRequest):
    """Extract data from a document image (base64 or URL)."""
    t0 = time.time()

    if req.image_base64:
        b64 = req.image_base64
    elif req.image_url:
        r = requests.get(req.image_url, timeout=30)
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        b64 = image_to_base64(img)
    else:
        raise HTTPException(400, "Provide image_base64 or image_url")

    prompt = resolve_prompt(req.prompt, req.prompt_template)
    raw = call_vlm(b64, prompt, req.max_tokens, req.temperature)
    parsed = parse_json_output(raw)

    return ExtractionResponse(
        status="ok",
        data=parsed,
        raw_output=raw,
        processing_time_ms=int((time.time() - t0) * 1000),
        prompt_used=prompt[:100] + "...",
    )

@app.post("/extract/upload", response_model=ExtractionResponse)
async def extract_from_upload(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
    prompt_template: Optional[str] = Form(None),
    max_tokens: int = Form(4096),
    temperature: float = Form(0.1),
):
    """Extract data from an uploaded document image."""
    t0 = time.time()

    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    b64 = image_to_base64(img)

    p = resolve_prompt(prompt, prompt_template)
    raw = call_vlm(b64, p, max_tokens, temperature)
    parsed = parse_json_output(raw)

    return ExtractionResponse(
        status="ok",
        data=parsed,
        raw_output=raw,
        processing_time_ms=int((time.time() - t0) * 1000),
        prompt_used=p[:100] + "...",
    )

@app.post("/extract/batch")
async def extract_batch(files: list[UploadFile] = File(...), prompt_template: Optional[str] = Form(None)):
    """Extract data from multiple document images."""
    results = []
    p = resolve_prompt(None, prompt_template)
    for file in files:
        t0 = time.time()
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        b64 = image_to_base64(img)
        raw = call_vlm(b64, p)
        parsed = parse_json_output(raw)
        results.append({
            "filename": file.filename,
            "status": "ok",
            "data": parsed,
            "raw_output": raw,
            "processing_time_ms": int((time.time() - t0) * 1000),
        })
    return {"results": results, "total": len(results)}
