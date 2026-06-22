"""
RunPod Serverless Handler — LoRA Training Pipeline.

Trains face LoRA models for escorts using ai-toolkit (Flux 1 Dev).
Receives training images as a ZIP, auto-captions with Claude Vision,
runs ai-toolkit training, uploads the result back to WordPress.

Input schema:
{
    "zip_url": str,              # URL to ZIP with training images
    "trigger_word": str,         # e.g. "escort_model"
    "training_steps": int,       # default 2000
    "lora_rank": int,            # default 32
    "resolution": int,           # default 1024
    "learning_rate": str,        # default "4e-5"
    "lora_type": str,            # "face" or "body"
    "caption_focus": str,        # "face" or "body"
    "hf_token": str,             # HuggingFace token (for model download)
    "lora_id": str,              # WordPress LoRA DB ID
    "callback_url": str,         # WordPress webhook URL
    "webhook_secret": str,       # Secret for webhook auth
    "anthropic_api_key": str,    # For Claude Vision captions
}

Output:
{
    "lora_id": int,
    "status": "ready" | "failed",
    "storage_key": str,          # URL to uploaded .safetensors
    "error": str                 # Only on failure
}
"""

import os
import io
import sys
import glob
import json
import time
import shutil
import zipfile
import base64
import subprocess
import traceback

try:
    import runpod
    print(f"[TGND-TRAIN] runpod {runpod.__version__}", flush=True)
except Exception as e:
    print(f"[TGND-TRAIN] FATAL: cannot import runpod: {e}", flush=True)
    sys.exit(1)

try:
    import requests
    from PIL import Image
    print("[TGND-TRAIN] requests + PIL OK", flush=True)
except Exception as e:
    print(f"[TGND-TRAIN] FATAL: cannot import requests/PIL: {e}", flush=True)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WORK_DIR = "/workspace"
TRAINING_DATA_DIR = os.path.join(WORK_DIR, "training_data")
OUTPUT_DIR = os.path.join(WORK_DIR, "output")
CONFIG_PATH = os.path.join(WORK_DIR, "train_config.yaml")
AI_TOOLKIT_DIR = "/app/ai-toolkit"

HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Caption defaults
MAX_CAPTION_RETRIES = 2
CAPTION_TIMEOUT = 30


# ---------------------------------------------------------------------------
# Claude Vision auto-captioning
# ---------------------------------------------------------------------------
def caption_image_claude(image_path, trigger_word, focus, anthropic_key):
    """Generate a caption for a training image using Claude Vision."""
    if not anthropic_key:
        return f"{trigger_word}, photo of a person"

    with open(image_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode("utf-8")

    ext = os.path.splitext(image_path)[1].lower()
    media_type = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"

    if focus == "body":
        prompt = (
            "Describe this photo of a person for AI training. Focus on body type, "
            "proportions, skin tone, and clothing. Be concise (1-2 sentences). "
            "Do NOT mention the face or identity. Do NOT mention the background unless relevant."
        )
    else:
        prompt = (
            "Describe this photo of a person for AI training. Focus on facial features, "
            "expression, hair, and overall appearance. Be concise (1-2 sentences). "
            "Do NOT mention the background unless relevant."
        )

    payload = {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 150,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": img_data,
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }],
    }

    for attempt in range(MAX_CAPTION_RETRIES + 1):
        try:
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": anthropic_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json=payload,
                timeout=CAPTION_TIMEOUT,
            )
            if resp.status_code == 200:
                data = resp.json()
                text = data["content"][0]["text"].strip()
                return f"{trigger_word}, {text}"
            else:
                print(f"[TGND-TRAIN] Caption API error {resp.status_code}: {resp.text[:200]}", flush=True)
        except Exception as e:
            print(f"[TGND-TRAIN] Caption request failed (attempt {attempt+1}): {e}", flush=True)

        if attempt < MAX_CAPTION_RETRIES:
            time.sleep(2)

    # Fallback
    return f"{trigger_word}, photo of a person"


def caption_all_images(data_dir, trigger_word, focus, anthropic_key):
    """Generate .txt caption files for all images in the directory."""
    image_exts = {".jpg", ".jpeg", ".png", ".webp"}
    images = sorted([
        f for f in os.listdir(data_dir)
        if os.path.splitext(f)[1].lower() in image_exts
    ])

    print(f"[TGND-TRAIN] Captioning {len(images)} images (focus={focus})...", flush=True)

    for i, img_name in enumerate(images):
        img_path = os.path.join(data_dir, img_name)
        txt_path = os.path.splitext(img_path)[0] + ".txt"

        # Skip if caption already exists (from ZIP)
        if os.path.exists(txt_path):
            print(f"  [{i+1}/{len(images)}] {img_name}: existing caption, skipping", flush=True)
            continue

        caption = caption_image_claude(img_path, trigger_word, focus, anthropic_key)
        with open(txt_path, "w") as f:
            f.write(caption)
        print(f"  [{i+1}/{len(images)}] {img_name}: {caption[:80]}...", flush=True)

    return len(images)


# ---------------------------------------------------------------------------
# ai-toolkit config
# ---------------------------------------------------------------------------
def write_training_config(trigger_word, steps, rank, lr, resolution, lora_type):
    """Write ai-toolkit YAML config for LoRA training."""
    name = f"tgnd_{lora_type}_{trigger_word}"

    # Sample prompts per type
    if lora_type == "body":
        sample_prompts = [
            f"photo of {trigger_word}, woman standing in sunlit apartment, natural light, warm tones",
            f"photo of {trigger_word}, full body shot, casual clothing, soft lighting",
        ]
    else:
        sample_prompts = [
            f"photo of {trigger_word}, portrait, soft natural lighting, shallow depth of field",
            f"photo of {trigger_word}, close-up face shot, warm golden hour light",
        ]

    config = {
        "job": "extension",
        "config": {
            "name": name,
            "process": [{
                "type": "sd_trainer",
                "training_folder": OUTPUT_DIR,
                "device": "cuda:0",
                "trigger_word": trigger_word,
                "network": {
                    "type": "lora",
                    "linear": rank,
                    "linear_alpha": rank,
                },
                "save": {
                    "dtype": "float16",
                    "save_every": 500,
                    "max_step_saves_to_keep": 2,
                    "push_to_hub": False,
                },
                "datasets": [{
                    "folder_path": TRAINING_DATA_DIR,
                    "caption_ext": "txt",
                    "caption_dropout_rate": 0.05,
                    "shuffle_tokens": False,
                    "cache_latents_to_disk": True,
                    "resolution": [resolution, resolution],
                }],
                "train": {
                    "batch_size": 1,
                    "steps": steps,
                    "gradient_accumulation_steps": 1,
                    "train_unet": True,
                    "train_text_encoder": False,
                    "gradient_checkpointing": True,
                    "noise_scheduler": "flowmatch",
                    "timestep_type": "weighted",
                    "optimizer": "adamw8bit",
                    "lr": float(lr),
                    "dtype": "bf16",
                },
                "model": {
                    "name_or_path": "black-forest-labs/FLUX.1-dev",
                    "is_flux": True,
                    "quantize": True,
                    "quantize_te": True,
                    "qtype": "qfloat8",
                    "low_vram": True,
                },
                "sample": {
                    "sampler": "flowmatch",
                    "sample_every": 500,
                    "width": 1024,
                    "height": 1024,
                    "prompts": sample_prompts,
                    "neg": "",
                    "seed": 42,
                    "walk_seed": True,
                    "guidance_scale": 3.5,
                    "sample_steps": 20,
                },
            }],
        },
    }

    import yaml
    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False)

    print(f"[TGND-TRAIN] Config written: {name}, steps={steps}, rank={rank}, lr={lr}, res={resolution}", flush=True)


# ---------------------------------------------------------------------------
# Upload LoRA — HuggingFace (primary) + WordPress (fallback)
# ---------------------------------------------------------------------------
HF_REPO_ID = "JulioIglesiass/tgnd-loras"


def upload_lora_to_huggingface(safetensors_path, lora_id, trigger_word, hf_token):
    """Upload .safetensors to HuggingFace and return the download URL."""
    from huggingface_hub import HfApi

    api = HfApi()
    filename = f"lora_{lora_id}_{trigger_word}.safetensors"
    size_mb = os.path.getsize(safetensors_path) / (1024 * 1024)

    print(f"[TGND-TRAIN] Uploading {filename} ({size_mb:.1f} MB) to HuggingFace...", flush=True)

    # Ensure repo exists
    try:
        api.create_repo(HF_REPO_ID, repo_type="model", private=True, token=hf_token, exist_ok=True)
    except Exception as e:
        print(f"[TGND-TRAIN] HF repo create: {e}", flush=True)

    api.upload_file(
        path_or_fileobj=safetensors_path,
        path_in_repo=filename,
        repo_id=HF_REPO_ID,
        repo_type="model",
        token=hf_token,
    )

    # Construct direct download URL
    storage_key = f"https://huggingface.co/{HF_REPO_ID}/resolve/main/{filename}"
    print(f"[TGND-TRAIN] HuggingFace upload OK: {storage_key}", flush=True)
    return storage_key


def upload_lora_to_wordpress(safetensors_path, lora_id, callback_url, webhook_secret):
    """Fallback: upload .safetensors to WordPress via REST API."""
    upload_url = callback_url.replace("/webhook", "/upload-lora")

    filename = os.path.basename(safetensors_path)
    file_size = os.path.getsize(safetensors_path) / (1024 * 1024)
    print(f"[TGND-TRAIN] WP fallback: uploading {filename} ({file_size:.1f} MB) to {upload_url}", flush=True)

    with open(safetensors_path, "rb") as f:
        resp = requests.post(
            upload_url,
            files={"lora_file": (filename, f, "application/octet-stream")},
            data={
                "lora_id": str(lora_id),
                "secret": webhook_secret,
            },
            timeout=600,
        )

    if resp.status_code == 200:
        data = resp.json()
        storage_key = data.get("storage_key", "")
        print(f"[TGND-TRAIN] WP upload OK: {storage_key}", flush=True)
        return storage_key
    else:
        raise Exception(f"WP upload failed ({resp.status_code}): {resp.text[:500]}")


def send_callback(callback_url, lora_id, status, storage_key, webhook_secret, error=None):
    """Send completion callback to WordPress."""
    payload = {
        "lora_id": lora_id,
        "status": status,
        "storage_key": storage_key,
        "secret": webhook_secret,
    }
    if error:
        payload["error"] = str(error)[:500]

    try:
        resp = requests.post(callback_url, json=payload, timeout=30)
        print(f"[TGND-TRAIN] Callback sent: status={status}, response={resp.status_code}", flush=True)
    except Exception as e:
        print(f"[TGND-TRAIN] Callback failed: {e}", flush=True)


# ---------------------------------------------------------------------------
# Main handler
# ---------------------------------------------------------------------------
def handler(job):
    """RunPod serverless handler for LoRA training."""
    inp = job.get("input", {})

    zip_url = inp.get("zip_url", "")
    trigger_word = inp.get("trigger_word", "escort_model")
    steps = int(inp.get("training_steps", 2000))
    rank = int(inp.get("lora_rank", 32))
    resolution = int(inp.get("resolution", 1024))
    lr = inp.get("learning_rate", "4e-5")
    lora_type = inp.get("lora_type", "face")
    caption_focus = inp.get("caption_focus", "face")
    hf_token = inp.get("hf_token", "") or HF_TOKEN
    lora_id = inp.get("lora_id", "0")
    callback_url = inp.get("callback_url", "")
    webhook_secret = inp.get("webhook_secret", "")
    anthropic_key = inp.get("anthropic_api_key", "")

    if not zip_url:
        return {"lora_id": lora_id, "status": "failed", "error": "No zip_url provided"}

    # Set HF token for model downloads
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

    try:
        # ── Step 1: Download and extract ZIP ──
        print(f"[TGND-TRAIN] Downloading training data from {zip_url[:80]}...", flush=True)

        for d in [TRAINING_DATA_DIR, OUTPUT_DIR]:
            if os.path.exists(d):
                shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)

        resp = requests.get(zip_url, timeout=120)
        resp.raise_for_status()

        zip_path = os.path.join(WORK_DIR, "training_data.zip")
        with open(zip_path, "wb") as f:
            f.write(resp.content)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(TRAINING_DATA_DIR)
        os.remove(zip_path)

        # Flatten: if ZIP contains a subdirectory, move files up
        subdirs = [d for d in os.listdir(TRAINING_DATA_DIR)
                    if os.path.isdir(os.path.join(TRAINING_DATA_DIR, d))]
        if len(subdirs) == 1 and not any(
            f for f in os.listdir(TRAINING_DATA_DIR)
            if os.path.isfile(os.path.join(TRAINING_DATA_DIR, f))
        ):
            subdir = os.path.join(TRAINING_DATA_DIR, subdirs[0])
            for item in os.listdir(subdir):
                shutil.move(os.path.join(subdir, item), TRAINING_DATA_DIR)
            os.rmdir(subdir)

        image_exts = {".jpg", ".jpeg", ".png", ".webp"}
        image_count = len([
            f for f in os.listdir(TRAINING_DATA_DIR)
            if os.path.splitext(f)[1].lower() in image_exts
        ])
        print(f"[TGND-TRAIN] Extracted {image_count} images", flush=True)

        if image_count == 0:
            raise Exception("No images found in ZIP")

        # ── Step 2: Auto-caption with Claude Vision ──
        caption_count = caption_all_images(
            TRAINING_DATA_DIR, trigger_word, caption_focus, anthropic_key
        )
        print(f"[TGND-TRAIN] Captioned {caption_count} images", flush=True)

        # ── Step 3: Write ai-toolkit config ──
        write_training_config(trigger_word, steps, rank, lr, resolution, lora_type)

        # ── Step 4: Login HuggingFace (needed for FLUX.1-dev download) ──
        if hf_token:
            try:
                from huggingface_hub import login
                login(token=hf_token)
                print("[TGND-TRAIN] HuggingFace login OK", flush=True)
            except Exception as e:
                print(f"[TGND-TRAIN] HuggingFace login warning: {e}", flush=True)

        # ── Step 5: Run training ──
        print(f"[TGND-TRAIN] Starting training: {steps} steps, rank {rank}...", flush=True)
        start_time = time.time()

        result = subprocess.run(
            ["python3", os.path.join(AI_TOOLKIT_DIR, "run.py"), CONFIG_PATH],
            cwd=AI_TOOLKIT_DIR,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour max
        )

        elapsed = time.time() - start_time
        print(f"[TGND-TRAIN] Training finished in {elapsed/60:.1f} min, exit code: {result.returncode}", flush=True)

        if result.returncode != 0:
            stderr = result.stderr[-2000:] if result.stderr else "no stderr"
            raise Exception(f"Training failed (exit {result.returncode}): {stderr}")

        # ── Step 6: Find the trained .safetensors ──
        safetensors = sorted(glob.glob(os.path.join(OUTPUT_DIR, "**/*.safetensors"), recursive=True))
        if not safetensors:
            raise Exception("Training completed but no .safetensors file found")

        latest = safetensors[-1]
        size_mb = os.path.getsize(latest) / (1024 * 1024)
        print(f"[TGND-TRAIN] LoRA file: {latest} ({size_mb:.1f} MB)", flush=True)

        # ── Step 7: Upload LoRA (HuggingFace primary, WP fallback) ──
        storage_key = ""
        if hf_token:
            try:
                storage_key = upload_lora_to_huggingface(latest, lora_id, trigger_word, hf_token)
            except Exception as e:
                print(f"[TGND-TRAIN] HF upload failed, trying WP: {e}", flush=True)

        if not storage_key and callback_url:
            storage_key = upload_lora_to_wordpress(latest, lora_id, callback_url, webhook_secret)

        # Send success callback to WordPress
        if callback_url:
            send_callback(callback_url, lora_id, "ready", storage_key, webhook_secret)

        output = {
            "lora_id": int(lora_id),
            "status": "ready",
            "storage_key": storage_key,
            "training_time_min": round(elapsed / 60, 1),
            "image_count": image_count,
        }
        print(f"[TGND-TRAIN] SUCCESS: {json.dumps(output)}", flush=True)
        return output

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"[TGND-TRAIN] FAILED: {error_msg}", flush=True)
        traceback.print_exc()

        # Send failure callback
        if callback_url and webhook_secret:
            send_callback(callback_url, lora_id, "failed", "", webhook_secret, error=error_msg)

        return {
            "lora_id": int(lora_id),
            "status": "failed",
            "error": error_msg,
        }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("[TGND-TRAIN] Handler starting...", flush=True)

    # Verify ai-toolkit is available
    try:
        result = subprocess.run(
            ["python3", "-c", "from toolkit.job import get_job; print('ai-toolkit OK')"],
            cwd=AI_TOOLKIT_DIR,
            capture_output=True,
            text=True,
            timeout=30,
        )
        print(f"[TGND-TRAIN] {result.stdout.strip()}", flush=True)
    except Exception as e:
        print(f"[TGND-TRAIN] WARNING: ai-toolkit check failed: {e}", flush=True)

    runpod.serverless.start({"handler": handler})
