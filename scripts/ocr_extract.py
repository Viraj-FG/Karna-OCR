"""
Karna OCR — Command-line document extraction tool.

Usage:
    python ocr_extract.py <image_path> [--prompt-template general|form|invoice|table|handwriting|contract]
    python ocr_extract.py <image_path> --prompt "Your custom prompt here"
    python ocr_extract.py <image_path> --output results.json
"""

import argparse, base64, io, json, sys, time
import requests
from PIL import Image

API_BASE = "http://localhost:8080"

def extract(image_path, prompt=None, prompt_template=None, output=None):
    img = Image.open(image_path).convert("RGB")
    print(f"Image: {image_path} ({img.width}x{img.height})")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    payload = {"image_base64": b64, "max_tokens": 4096, "temperature": 0.1}
    if prompt:
        payload["prompt"] = prompt
    elif prompt_template:
        payload["prompt_template"] = prompt_template

    print(f"Extracting (template: {prompt_template or 'general'})...")
    t0 = time.time()
    resp = requests.post(f"{API_BASE}/extract", json=payload, timeout=180)
    elapsed = time.time() - t0

    if resp.status_code != 200:
        print(f"Error {resp.status_code}: {resp.text}")
        sys.exit(1)

    result = resp.json()
    print(f"Done in {elapsed:.1f}s ({result.get('processing_time_ms', 0)}ms server-side)\n")

    if result.get("data"):
        print(json.dumps(result["data"], indent=2, ensure_ascii=False))
    else:
        print(result.get("raw_output", "No output"))

    if output:
        with open(output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to {output}")

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Karna OCR — Extract data from documents")
    parser.add_argument("image", help="Path to document image")
    parser.add_argument("--prompt", help="Custom extraction prompt")
    parser.add_argument("--prompt-template", choices=["general", "form", "invoice", "table", "handwriting", "contract"], default=None)
    parser.add_argument("--output", "-o", help="Save full result to JSON file")
    args = parser.parse_args()

    extract(args.image, prompt=args.prompt, prompt_template=args.prompt_template, output=args.output)
