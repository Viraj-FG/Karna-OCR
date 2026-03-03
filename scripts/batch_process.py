"""
Karna OCR — Batch document processor.

Usage:
    python batch_process.py /path/to/documents/ --output results/
    python batch_process.py /path/to/documents/ --prompt-template invoice --output results/
"""

import argparse, json, os, sys, time
from pathlib import Path
import requests

API_BASE = "http://localhost:8080"
SUPPORTED = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".pdf"}

def process_batch(input_dir, output_dir, prompt_template=None):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    files = sorted([f for f in input_path.iterdir() if f.suffix.lower() in SUPPORTED])
    print(f"Found {len(files)} documents in {input_dir}\n")

    results = []
    for i, f in enumerate(files):
        print(f"[{i+1}/{len(files)}] {f.name}...", end=" ", flush=True)
        t0 = time.time()

        with open(f, "rb") as fh:
            form_data = {"file": (f.name, fh, "application/octet-stream")}
            params = {}
            if prompt_template:
                params["prompt_template"] = (None, prompt_template)
            resp = requests.post(f"{API_BASE}/extract/upload", files=form_data, data={"prompt_template": prompt_template or ""}, timeout=180)

        elapsed = time.time() - t0

        if resp.status_code == 200:
            result = resp.json()
            out_file = output_path / f"{f.stem}.json"
            with open(out_file, "w", encoding="utf-8") as of:
                json.dump(result, of, indent=2, ensure_ascii=False)
            print(f"OK ({elapsed:.1f}s)")
            results.append({"file": f.name, "status": "ok", "time": elapsed})
        else:
            print(f"FAIL ({resp.status_code})")
            results.append({"file": f.name, "status": "error", "error": resp.text[:200]})

    # Summary
    ok = sum(1 for r in results if r["status"] == "ok")
    total_time = sum(r.get("time", 0) for r in results)
    print(f"\n{'='*50}")
    print(f"Processed: {ok}/{len(results)} documents")
    print(f"Total time: {total_time:.1f}s (avg {total_time/max(len(results),1):.1f}s/doc)")
    print(f"Results saved to: {output_dir}/")

    with open(output_path / "_summary.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Karna OCR — Batch document extraction")
    parser.add_argument("input_dir", help="Directory containing document images")
    parser.add_argument("--output", "-o", default="./results", help="Output directory")
    parser.add_argument("--prompt-template", choices=["general", "form", "invoice", "table", "handwriting", "contract"])
    args = parser.parse_args()

    process_batch(args.input_dir, args.output, args.prompt_template)
