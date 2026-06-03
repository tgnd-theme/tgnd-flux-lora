"""
One-time script: download Flux Dev model to RunPod network volume.
Run this as a RunPod pod with the network volume mounted at /runpod-volume.
"""

import os
import time

print("[SETUP] Starting Flux Dev model download to network volume...")
t0 = time.time()

# Login to HuggingFace
hf_token = os.environ.get("HF_TOKEN", "")
if hf_token:
    from huggingface_hub import login
    login(token=hf_token)
    print("[SETUP] Logged in to HuggingFace")

target = "/runpod-volume/flux-dev"

if os.path.exists(target) and os.listdir(target):
    print(f"[SETUP] Model already exists at {target}, skipping download")
else:
    print("[SETUP] Downloading Flux Dev from HuggingFace Hub...")
    from huggingface_hub import snapshot_download
    snapshot_download(
        "black-forest-labs/FLUX.1-dev",
        local_dir=target,
        local_dir_use_symlinks=False,
    )
    elapsed = time.time() - t0

    # Check size
    total = 0
    for root, dirs, files in os.walk(target):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))

    print(f"[SETUP] Done! {total / 1024**3:.1f}GB downloaded in {elapsed:.0f}s")

# Also pre-download the LoRAs
import requests

loras = {
    "face": "https://v3b.fal.media/files/b/0a9cae35/F1eD2zOe2VXJ1tQKWIjTo_pytorch_lora_weights.safetensors",
    "style": "https://v3b.fal.media/files/b/0a9cce1b/5Dl5QdFUy2SHFdFqA9WDh_pytorch_lora_weights.safetensors",
}

lora_dir = "/runpod-volume/loras"
os.makedirs(lora_dir, exist_ok=True)

for name, url in loras.items():
    path = f"{lora_dir}/{name}.safetensors"
    if os.path.exists(path):
        print(f"[SETUP] LoRA {name} already cached")
        continue
    print(f"[SETUP] Downloading LoRA: {name}...")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    with open(path, "wb") as f:
        f.write(resp.content)
    print(f"[SETUP] LoRA {name}: {len(resp.content) / 1024**2:.1f}MB")

print("[SETUP] All done! Volume is ready.")
