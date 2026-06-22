#!/usr/bin/env python3
"""
Batch generate lookbook photos via RunPod Ultimate endpoint.

Uses all 50 lookbook references with babe's triple LoRA (face+body+style).
Saves results to ~/Desktop/The Girl Next Door/zishy/clean/
"""

import json
import os
import sys
import time
import base64
import random
import requests

# ─── Config ───
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")
ENDPOINT_ID = "cgvk5tmqgtdm6h"
RUNPOD_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"

STAGING_BASE = "https://staging.the-girl-next-door.com/wp-content/uploads/tgnd-studio"
LOOKBOOK_BASE = "https://huggingface.co/JulioIglesiass/tgnd-lookbook/resolve/main/clean"
LORA_BASE = f"{STAGING_BASE}/loras"
FACE_REF_BASE = "https://huggingface.co/JulioIglesiass/tgnd-loras/resolve/main/face_refs"

# ─── Escort Profiles ───
# Each escort has their own LoRAs, body description, and face refs for PuLID
ESCORTS = {
    "agatha": {
        "loras": [
            {"url": f"{LORA_BASE}/lora_11_agatha_model.safetensors", "scale": 0.4, "trigger": "agatha_model"},
            {"url": f"{LORA_BASE}/zishy_style_aitk.safetensors", "scale": 0.5, "trigger": "zishy_style"},
        ],
        "body_desc": "curvy Latina woman, olive tan skin, natural C-cup breasts, black hair, toned body",
        "face_refs": [
            f"{FACE_REF_BASE}/agatha/HSsI-agatha-duarte.jpeg",
            f"{FACE_REF_BASE}/agatha/Scy4-agatha-duarte.jpeg",
            f"{FACE_REF_BASE}/agatha/tTU2-agatha-duarte.jpeg",
        ],
        "pulid_strength": 0.8,
    },
    "babe": {
        "loras": [
            {"url": f"{LORA_BASE}/babe_face_v2_aitk.safetensors", "scale": 0.5, "trigger": "babe_model"},
            {"url": f"{LORA_BASE}/zishy_style_aitk.safetensors", "scale": 0.5, "trigger": "zishy_style"},
        ],
        "body_desc": "slim petite Latina woman, olive tan skin, small natural A-cup breasts, dark brown hair with caramel highlights, diamond stud earrings, toned flat stomach",
        "face_refs": [],  # No face refs yet for babe
        "pulid_strength": 0.8,
    },
}

# Default escort (can override via --escort flag)
DEFAULT_ESCORT = "agatha"

CATALOG_PATH = os.path.expanduser("~/Desktop/lookbook/catalog.json")
OUTPUT_DIR = os.path.expanduser("~/Desktop/The Girl Next Door/zishy/clean")

HEADERS = {
    "Authorization": f"Bearer {RUNPOD_API_KEY}",
    "Content-Type": "application/json",
}


def build_prompt(entry, escort_profile):
    """Build generation prompt from catalog entry."""
    desc = entry.get("description", "")
    category = entry.get("category", "")
    pose = entry.get("pose", "standing")
    mood = entry.get("mood", "warm")
    topless = entry.get("topless", False)

    # LoRA triggers from escort profile
    triggers = " ".join(l["trigger"] for l in escort_profile["loras"])
    body_desc = escort_profile["body_desc"]

    # Base prompt with LoRA triggers
    parts = [
        triggers,
        f"a beautiful {body_desc}",
    ]

    # Expression
    parts.append("smiling playfully, natural expression")

    # Scene + clothing from description
    if desc:
        parts.append(desc)

    # Clothing override for topless
    if topless:
        parts.append("topless, bare breasts, casual bottom")
    elif entry.get("clothing") == "underwear":
        parts.append("wearing white cotton top and thong")
    elif entry.get("clothing") == "casual":
        parts.append("wearing casual outfit")
    elif entry.get("clothing") == "lingerie":
        parts.append("wearing delicate lace lingerie set")

    # Scene mood
    parts.append(f"{mood} natural lighting, candid Zishy photography style")

    # Anatomy safety
    parts.append("two arms, two legs, correct human anatomy, five fingers on each hand")

    return ", ".join(parts)


def build_clothing_desc(entry):
    """Clothing description for fix passes."""
    if entry.get("topless", False):
        return "topless, bare breasts"
    clothing = entry.get("clothing", "casual")
    mapping = {
        "casual": "casual outfit, top and shorts",
        "underwear": "white cotton top and thong underwear",
        "lingerie": "delicate lace lingerie set",
        "topless": "topless, bare breasts",
    }
    return mapping.get(clothing, clothing)


def submit_job(entry, escort_profile, seed=None):
    """Submit a generation job to RunPod."""
    if seed is None:
        seed = random.randint(1, 2147483647)

    payload = {
        "input": {
            "lookbook_image_url": f"{LOOKBOOK_BASE}/{entry['file']}",
            "prompt": build_prompt(entry, escort_profile),
            "clothing_prompt": "",
            "clothing_desc": build_clothing_desc(entry),
            "expression_hint": "smiling, playful",
            "loras": escort_profile["loras"],
            "body_description": escort_profile["body_desc"],
            "face_reference_urls": escort_profile.get("face_refs", []),
            "pulid_strength": escort_profile.get("pulid_strength", 0.8),
            "strength": 0.80,
            "guidance": 5.0,
            "seed": seed,
            "skip_fix": True,
            "no_filmgrade": False,
        }
    }

    resp = requests.post(f"{RUNPOD_URL}/run", headers=HEADERS, json=payload)
    resp.raise_for_status()
    data = resp.json()
    job_id = data.get("id")
    print(f"  Submitted job {job_id} for {entry['file']} (seed={seed})")
    return job_id, seed


def poll_job(job_id, timeout=600, interval=5):
    """Poll a job until completion."""
    start = time.time()
    while time.time() - start < timeout:
        resp = requests.get(f"{RUNPOD_URL}/status/{job_id}", headers=HEADERS)
        if resp.status_code == 404:
            time.sleep(interval)
            continue
        data = resp.json()
        status = data.get("status", "UNKNOWN")

        if status == "COMPLETED":
            return data.get("output", {})
        elif status in ("FAILED", "CANCELLED", "TIMED_OUT"):
            print(f"  Job {job_id} failed: {status}")
            return None
        elif status in ("IN_QUEUE", "IN_PROGRESS"):
            elapsed = time.time() - start
            print(f"  [{elapsed:.0f}s] Job {job_id}: {status}", end="\r")
            time.sleep(interval)
        else:
            time.sleep(interval)

    print(f"  Job {job_id} timed out after {timeout}s")
    return None


def save_image(b64_data, filename):
    """Save base64 JPEG to disk."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64_data))
    print(f"  Saved: {path}")
    return path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Batch generate lookbook photos")
    parser.add_argument("--escort", default=DEFAULT_ESCORT, choices=ESCORTS.keys(),
                        help=f"Escort to generate for (default: {DEFAULT_ESCORT})")
    parser.add_argument("--limit", type=int, default=0, help="Max photos to generate (0=all)")
    args = parser.parse_args()

    escort_profile = ESCORTS[args.escort]
    escort_name = args.escort

    # Load catalog
    with open(CATALOG_PATH) as f:
        catalog = json.load(f)

    if args.limit:
        catalog = catalog[:args.limit]

    print(f"Escort: {escort_name}")
    print(f"LoRAs: {[l['trigger'] for l in escort_profile['loras']]}")
    print(f"PuLID face refs: {len(escort_profile.get('face_refs', []))}")
    print(f"Loaded {len(catalog)} lookbook references")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Endpoint: {ENDPOINT_ID}")
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Check which ones are already done
    existing = set(os.listdir(OUTPUT_DIR)) if os.path.exists(OUTPUT_DIR) else set()

    # First, send a health check to warm up the endpoint
    print("Sending warm-up health check...")
    try:
        warmup = requests.post(f"{RUNPOD_URL}/run", headers=HEADERS, json={
            "input": {"health_check": True}
        })
        warmup_data = warmup.json()
        warmup_id = warmup_data.get("id")
        print(f"  Warm-up job: {warmup_id}")
        # Wait for warm-up
        result = poll_job(warmup_id, timeout=600, interval=10)
        if result and result.get("status") == "ok":
            print(f"  Endpoint warm and ready! VRAM: {result.get('vram_gb', '?')}GB")
        else:
            print(f"  Warm-up result: {result}")
            print("  Proceeding anyway...")
    except Exception as e:
        print(f"  Warm-up failed: {e}")
        print("  Proceeding anyway...")

    print()

    # Submit jobs in batches of 3 (max workers)
    BATCH_SIZE = 3
    results_summary = []

    for batch_start in range(0, len(catalog), BATCH_SIZE):
        batch = catalog[batch_start:batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(catalog) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"\n{'='*60}")
        print(f"Batch {batch_num}/{total_batches} ({len(batch)} photos)")
        print(f"{'='*60}")

        # Check for existing outputs
        jobs = []
        for entry in batch:
            base = os.path.splitext(entry["file"])[0]
            out_name = f"ult_{base}.jpg"
            if out_name in existing:
                print(f"  SKIP {entry['file']} (already exists)")
                results_summary.append({"file": entry["file"], "status": "skipped"})
                continue

            job_id, seed = submit_job(entry, escort_profile)
            jobs.append({"job_id": job_id, "entry": entry, "seed": seed, "out_name": out_name})

        # Poll all jobs in batch
        for job_info in jobs:
            job_id = job_info["job_id"]
            entry = job_info["entry"]
            print(f"\n  Waiting for {entry['file']}...")
            output = poll_job(job_id, timeout=600, interval=5)

            if output and output.get("status") == "ok":
                save_image(output["image"], job_info["out_name"])
                elapsed = output.get("inference_time", 0)
                passes = output.get("passes_run", [])
                version = output.get("handler_version", "?")
                skin_debug = output.get("skin_debug", None)
                skin_info = f", skin_debug={skin_debug}" if skin_debug else ""
                print(f"  Done! {elapsed:.1f}s, passes: {', '.join(passes)}, handler={version}{skin_info}")
                results_summary.append({
                    "file": entry["file"],
                    "status": "ok",
                    "seed": job_info["seed"],
                    "time": elapsed,
                    "passes": passes,
                })
            else:
                error = output.get("error", "unknown") if output else "timeout/failed"
                print(f"  FAILED: {error}")
                results_summary.append({
                    "file": entry["file"],
                    "status": "failed",
                    "error": error,
                })

    # Summary
    print(f"\n\n{'='*60}")
    print("GENERATION COMPLETE")
    print(f"{'='*60}")
    ok = sum(1 for r in results_summary if r["status"] == "ok")
    failed = sum(1 for r in results_summary if r["status"] == "failed")
    skipped = sum(1 for r in results_summary if r["status"] == "skipped")
    print(f"  OK: {ok}, Failed: {failed}, Skipped: {skipped}")

    # Save summary
    summary_path = os.path.join(OUTPUT_DIR, "_generation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()
