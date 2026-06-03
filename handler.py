"""
RunPod Serverless Handler — Flux Dev + Multi-LoRA inference.

No safety checker. No NSFW filter. Full control.

Input:
{
    "prompt": "...",
    "loras": [                                  # optional, list of LoRAs
        {"url": "https://...safetensors", "scale": 1.0},
        {"url": "https://...safetensors", "scale": 0.8}
    ],
    "lora_url": "https://...safetensors",       # legacy single LoRA support
    "lora_scale": 1.0,
    "width": 768,
    "height": 1024,
    "guidance_scale": 2.5,
    "num_inference_steps": 28,
    "seed": 42                                  # optional, random if omitted
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
import hashlib
import random
import time
import runpod
import torch
from diffusers import FluxPipeline
import requests

# ─── Globals (loaded once on cold start) ───
pipe = None
loaded_lora_key = None


def load_model():
    """Load Flux Dev pipeline once. Tries network volume first, then HF Hub."""
    global pipe
    if pipe is not None:
        return

    print("[TGND] Loading Flux Dev pipeline...")
    t0 = time.time()

    volume_path = "/runpod-volume/flux-dev"
    if os.path.exists(volume_path):
        print(f"[TGND] Loading from network volume: {volume_path}")
        model_source = volume_path
    else:
        hf_token = os.environ.get("HF_TOKEN", "")
        if hf_token:
            from huggingface_hub import login
            login(token=hf_token)
        model_source = "black-forest-labs/FLUX.1-dev"
        print("[TGND] Downloading from HuggingFace Hub...")

    pipe = FluxPipeline.from_pretrained(
        model_source,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()

    print(f"[TGND] Pipeline loaded in {time.time() - t0:.1f}s")


def download_lora(url):
    """Download LoRA file, using cache if available."""
    url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
    cache_path = f"/tmp/lora_{url_hash}.safetensors"

    if os.path.exists(cache_path):
        print(f"[TGND] LoRA cached: {cache_path}")
        return cache_path

    print(f"[TGND] Downloading LoRA: {url[:80]}...")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    with open(cache_path, "wb") as f:
        f.write(resp.content)
    print(f"[TGND] Downloaded {len(resp.content) / 1024 / 1024:.1f}MB")
    return cache_path


def load_loras(lora_configs):
    """Load one or more LoRAs. Caches the combination."""
    global loaded_lora_key

    if not lora_configs:
        # No LoRAs requested — unload if any loaded
        if loaded_lora_key:
            try:
                pipe.unload_lora_weights()
            except Exception:
                pass
            loaded_lora_key = None
        return None

    # Create a cache key from URLs
    cache_key = "|".join(sorted(c["url"] for c in lora_configs))
    if loaded_lora_key == cache_key:
        print(f"[TGND] LoRAs already loaded ({len(lora_configs)} adapters)")
        return {f"lora_{i}": c["scale"] for i, c in enumerate(lora_configs)}

    # Unload previous
    try:
        pipe.unload_lora_weights()
    except Exception:
        pass

    t0 = time.time()
    adapter_names = []
    adapter_weights = []

    for i, cfg in enumerate(lora_configs):
        name = f"lora_{i}"
        path = download_lora(cfg["url"])
        pipe.load_lora_weights(path, adapter_name=name)
        adapter_names.append(name)
        adapter_weights.append(cfg["scale"])
        print(f"[TGND] Loaded adapter '{name}' (scale={cfg['scale']})")

    # Set active adapters with weights
    if len(adapter_names) > 1:
        pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)

    loaded_lora_key = cache_key
    print(f"[TGND] {len(lora_configs)} LoRAs loaded in {time.time() - t0:.1f}s")

    # Return scale for joint_attention_kwargs (use max scale)
    return {name: weight for name, weight in zip(adapter_names, adapter_weights)}


def handler(job):
    """RunPod serverless handler."""
    inp = job.get("input", {})

    prompt = inp.get("prompt", "")
    if not prompt:
        return {"status": "error", "error": "prompt is required"}

    # Load model on first request
    load_model()

    # Build LoRA config list
    lora_configs = inp.get("loras", [])
    if not lora_configs:
        # Legacy single LoRA support
        lora_url = inp.get("lora_url", "")
        if lora_url:
            lora_configs = [{"url": lora_url, "scale": float(inp.get("lora_scale", 1.0))}]

    adapters = load_loras(lora_configs)

    # Generation params
    width = int(inp.get("width", 768))
    height = int(inp.get("height", 1024))
    guidance_scale = float(inp.get("guidance_scale", 2.5))
    num_steps = int(inp.get("num_inference_steps", 28))
    seed = int(inp.get("seed", random.randint(1, 2147483647)))

    print(f"[TGND] Generating {width}x{height}, steps={num_steps}, guidance={guidance_scale}, seed={seed}")
    t0 = time.time()

    generator = torch.Generator("cuda").manual_seed(seed)

    # Use scale from first adapter if single LoRA
    lora_scale = lora_configs[0]["scale"] if len(lora_configs) == 1 else 1.0

    result = pipe(
        prompt=prompt,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=num_steps,
        generator=generator,
        joint_attention_kwargs={"scale": lora_scale} if lora_configs else None,
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
