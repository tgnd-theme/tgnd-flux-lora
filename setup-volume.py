"""
One-time script: download Flux 2 Dev (4-bit) model + YOLO models to RunPod network volume.
Run this as a RunPod pod with the network volume mounted at /runpod-volume.
"""

import os
import time

print("[SETUP] Starting Flux 2 Dev model download to network volume...")
t0 = time.time()

# Login to HuggingFace (required — FLUX.2-dev is gated)
hf_token = os.environ.get("HF_TOKEN", "")
if hf_token:
    from huggingface_hub import login
    login(token=hf_token)
    print("[SETUP] Logged in to HuggingFace")
else:
    print("[SETUP] WARNING: No HF_TOKEN set — FLUX.2-dev is gated, download may fail!")

# ─── Flux 2 Dev (4-bit quantized) ───
target = "/runpod-volume/flux2-dev"

if os.path.exists(target) and os.listdir(target):
    print(f"[SETUP] Model already exists at {target}, skipping download")
else:
    print("[SETUP] Downloading Flux 2 Dev (4-bit) from HuggingFace Hub...")
    from huggingface_hub import snapshot_download
    snapshot_download(
        "diffusers/FLUX.2-dev-bnb-4bit",
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

# ─── LoRAs (Flux 1 LoRAs — will need retraining for Flux 2) ───
import requests

loras = {
    "face": "https://v3b.fal.media/files/b/0a9cae35/F1eD2zOe2VXJ1tQKWIjTo_pytorch_lora_weights.safetensors",
    "style_v2": "https://v3b.fal.media/files/b/0a9cd2fa/R8JD6ub169gxMZNBnmX8n_pytorch_lora_weights.safetensors",
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

# ─── YOLO models for ADetailer ───
from huggingface_hub import hf_hub_download

yolo_dir = "/runpod-volume/yolo"
os.makedirs(yolo_dir, exist_ok=True)

yolo_models = ["face_yolov9c.pt", "hand_yolov9c.pt"]
for model_name in yolo_models:
    path = os.path.join(yolo_dir, model_name)
    if os.path.exists(path):
        print(f"[SETUP] YOLO {model_name} already cached")
        continue
    print(f"[SETUP] Downloading YOLO model: {model_name}...")
    hf_hub_download("Bingsu/adetailer", model_name, local_dir=yolo_dir)
    print(f"[SETUP] YOLO {model_name} downloaded")

print("[SETUP] All done! Volume is ready for Flux 2 Dev + ADetailer.")
