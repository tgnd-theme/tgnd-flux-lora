"""
RunPod Serverless Handler — Flux Dev + LoRA inference.

No safety checker. No NSFW filter. Full control.

Input:
{
    "prompt": "...",
    "lora_url": "https://...safetensors",   # optional, cached after first load
    "lora_scale": 1.0,                       # optional, default 1.0
    "width": 768,
    "height": 1024,
    "guidance_scale": 1.5,
    "num_inference_steps": 28,
    "seed": 42                               # optional, random if omitted
}

Output:
{
    "status": "ok",
    "image": "<base64 JPEG>",
    "seed": 42
}
"""

import os
import io
import base64
import random
import time
import runpod
import torch
from diffusers import FluxPipeline
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
import requests

# ─── Globals (loaded once on cold start) ───
pipe = None
loaded_lora_url = None


def load_model():
    """Load Flux Dev pipeline once. Tries network volume first, then HF Hub."""
    global pipe
    if pipe is not None:
        return

    print("[TGND] Loading Flux Dev pipeline...")
    t0 = time.time()

    # Try network volume first (fast, no download)
    volume_path = "/runpod-volume/flux-dev"
    if os.path.exists(volume_path):
        print(f"[TGND] Loading from network volume: {volume_path}")
        model_source = volume_path
    else:
        # Fallback to HuggingFace Hub (requires HF_TOKEN env var)
        hf_token = os.environ.get("HF_TOKEN", "")
        if hf_token:
            from huggingface_hub import login
            login(token=hf_token)
        model_source = "black-forest-labs/FLUX.1-dev"
        print(f"[TGND] Downloading from HuggingFace Hub...")

    pipe = FluxPipeline.from_pretrained(
        model_source,
        torch_dtype=torch.bfloat16,
    )

    # Enable memory optimizations for GPU
    pipe.enable_model_cpu_offload()

    print(f"[TGND] Pipeline loaded in {time.time() - t0:.1f}s")


def load_lora(lora_url, lora_scale=1.0):
    """Load LoRA weights from URL, cached between requests."""
    global loaded_lora_url

    if not lora_url:
        return

    if loaded_lora_url == lora_url:
        print(f"[TGND] LoRA already loaded: {lora_url[:60]}...")
        return

    print(f"[TGND] Loading LoRA from: {lora_url[:80]}...")
    t0 = time.time()

    # Download to temp file
    lora_path = "/tmp/current_lora.safetensors"
    resp = requests.get(lora_url, timeout=120)
    resp.raise_for_status()
    with open(lora_path, "wb") as f:
        f.write(resp.content)

    # Unload any previous LoRA
    try:
        pipe.unload_lora_weights()
    except Exception:
        pass

    # Load new LoRA
    pipe.load_lora_weights(lora_path)

    loaded_lora_url = lora_url
    print(f"[TGND] LoRA loaded in {time.time() - t0:.1f}s ({len(resp.content) / 1024 / 1024:.1f}MB)")


def handler(job):
    """RunPod serverless handler."""
    inp = job.get("input", {})

    prompt = inp.get("prompt", "")
    if not prompt:
        return {"status": "error", "error": "prompt is required"}

    # Load model on first request
    load_model()

    # Load LoRA if specified
    lora_url = inp.get("lora_url", "")
    lora_scale = float(inp.get("lora_scale", 1.0))
    if lora_url:
        load_lora(lora_url, lora_scale)

    # Generation params
    width = int(inp.get("width", 768))
    height = int(inp.get("height", 1024))
    guidance_scale = float(inp.get("guidance_scale", 1.5))
    num_steps = int(inp.get("num_inference_steps", 28))
    seed = int(inp.get("seed", random.randint(1, 2147483647)))

    print(f"[TGND] Generating {width}x{height}, steps={num_steps}, guidance={guidance_scale}, seed={seed}")
    t0 = time.time()

    generator = torch.Generator("cuda").manual_seed(seed)

    result = pipe(
        prompt=prompt,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=num_steps,
        generator=generator,
        joint_attention_kwargs={"scale": lora_scale} if lora_url else None,
    )

    image = result.images[0]

    # Encode to JPEG base64
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=92)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    elapsed = time.time() - t0
    print(f"[TGND] Generated in {elapsed:.1f}s, size={len(b64) // 1024}KB")

    return {
        "status": "ok",
        "image": b64,
        "seed": seed,
        "inference_time": round(elapsed, 2),
    }


# ─── RunPod entry point ───
runpod.serverless.start({"handler": handler})
