"""
RunPod Serverless Handler — Flux 2 Dev + Multi-LoRA + ADetailer inference.

No safety checker. No NSFW filter. Full control.
"""

import os
import io
import sys
import base64
import hashlib
import random
import time
import traceback

# Wrap all imports in try/except to diagnose crashes
try:
    import runpod
    print(f"[TGND] runpod {runpod.__version__}", flush=True)
except Exception as e:
    print(f"[TGND] FATAL: cannot import runpod: {e}", flush=True)
    sys.exit(1)

try:
    import torch
    print(f"[TGND] torch {torch.__version__}, CUDA: {torch.cuda.is_available()}", flush=True)
except Exception as e:
    print(f"[TGND] FATAL: cannot import torch: {e}", flush=True)
    sys.exit(1)

try:
    import numpy as np
    from PIL import Image, ImageFilter
    print(f"[TGND] numpy {np.__version__}, PIL OK", flush=True)
except Exception as e:
    print(f"[TGND] FATAL: cannot import numpy/PIL: {e}", flush=True)
    sys.exit(1)

# ─── Globals (loaded once on cold start) ───
pipe = None
inpaint_pipe = None
img2img_pipe = None
loaded_lora_key = None
yolo_face = None
yolo_hand = None


def load_model():
    """Load Flux 2 Dev pipeline once (4-bit quantized via bitsandbytes).
    Tries network volume first, then HF Hub."""
    global pipe
    if pipe is not None:
        return

    print("[TGND] Loading Flux 2 Dev pipeline...", flush=True)
    t0 = time.time()

    from diffusers import Flux2Pipeline, BitsAndBytesConfig, PipelineQuantizationConfig

    model_id = "black-forest-labs/FLUX.2-dev"
    volume_path = "/runpod-volume/flux2-dev-nf4"
    print(f"[TGND] Using Flux2Pipeline (Flux 2 Dev), model={model_id}", flush=True)

    volume_mounted = os.path.exists("/runpod-volume")

    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)

    # Always load from HF Hub with NF4 quantization (quantized models can't be saved/loaded from volume)
    print("[TGND] Loading Flux 2 Dev with NF4 quantization from HF Hub...", flush=True)

    # Quantize transformer only (biggest component) — NF4 reduces ~60GB to ~16GB
    quant_config = PipelineQuantizationConfig(
        quant_backend="bitsandbytes_4bit",
        quant_kwargs={
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.bfloat16,
        },
        components_to_quantize=["transformer"],
    )
    pipe = Flux2Pipeline.from_pretrained(
        model_id,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
    )

    print(f"[TGND] Pipeline loaded in {time.time() - t0:.1f}s", flush=True)


def load_inpaint_pipe():
    """Create inpaint pipeline from the main pipeline (shares weights, no extra VRAM)."""
    global inpaint_pipe
    if inpaint_pipe is not None:
        return

    try:
        # Try Flux 2 inpaint first, fall back to Flux 1
        try:
            from diffusers import Flux2InpaintPipeline
            inpaint_pipe = Flux2InpaintPipeline.from_pipe(pipe)
        except ImportError:
            from diffusers import FluxInpaintPipeline
            inpaint_pipe = FluxInpaintPipeline.from_pipe(pipe)
        print("[TGND] Inpaint pipeline created from main pipe (shared weights)", flush=True)
    except Exception as e:
        print(f"[TGND] Could not create inpaint pipeline: {e}", flush=True)
        inpaint_pipe = False  # sentinel: tried and failed


def prepare_img2img_latents(image, strength, num_steps, width, height, generator):
    """Prepare noisy latents from an input image for Flux 2 img2img.
    Uses flow matching: z_t = (1-t) * z_0 + t * noise."""
    vae = pipe.vae

    # Use pipeline's image processor if available for proper preprocessing
    if hasattr(pipe, 'image_processor'):
        img_tensor = pipe.image_processor.preprocess(image, height=height, width=width)
        img_tensor = img_tensor.to(device=vae.device, dtype=vae.dtype)
    else:
        img_tensor = torch.from_numpy(np.array(image)).float() / 255.0
        img_tensor = img_tensor * 2.0 - 1.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.to(device=vae.device, dtype=vae.dtype)

    print(f"[TGND] img2img input tensor: {img_tensor.shape}", flush=True)

    # Encode to latent space
    with torch.no_grad():
        encoded = vae.encode(img_tensor)

    print(f"[TGND] VAE encode output type: {type(encoded).__name__}", flush=True)

    # Extract latents from various output types
    if hasattr(encoded, 'latent_dist'):
        latents = encoded.latent_dist.sample(generator)
        print(f"[TGND] Used latent_dist.sample(): {latents.shape}", flush=True)
    elif hasattr(encoded, 'latents'):
        latents = encoded.latents
        print(f"[TGND] Used .latents: {latents.shape}", flush=True)
    elif hasattr(encoded, 'sample'):
        # Some VAEs return an object with .sample attribute (not method)
        s = encoded.sample
        latents = s() if callable(s) else s
        print(f"[TGND] Used .sample: {latents.shape}", flush=True)
    elif isinstance(encoded, (tuple, list)):
        latents = encoded[0]
        print(f"[TGND] Used tuple[0]: {latents.shape}", flush=True)
    else:
        latents = encoded
        print(f"[TGND] Used raw: {type(latents).__name__}, shape={getattr(latents, 'shape', '?')}", flush=True)

    # Scale
    sf = getattr(vae.config, 'scaling_factor', None)
    if sf:
        latents = latents * sf
        print(f"[TGND] Scaled by {sf}", flush=True)

    shift = getattr(vae.config, 'shift_factor', None)
    if shift:
        latents = latents - shift
        print(f"[TGND] Shifted by {shift}", flush=True)

    print(f"[TGND] Final latents: shape={latents.shape}, ndim={latents.ndim}", flush=True)

    # Ensure 4D [B, C, H, W]
    while latents.ndim < 4:
        latents = latents.unsqueeze(0)

    batch, channels, h, w = latents.shape
    print(f"[TGND] Unpacked: batch={batch}, channels={channels}, h={h}, w={w}", flush=True)

    # Pack latents into Flux format: [B, seq_len, C*4] using 2x2 patches
    latents = latents.reshape(batch, channels, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch, (h // 2) * (w // 2), channels * 4)

    # Flow matching: mix clean latents with noise
    noise = torch.randn_like(latents)
    noisy_latents = (1.0 - strength) * latents + strength * noise

    print(f"[TGND] Packed img2img latents: {noisy_latents.shape}, strength={strength}", flush=True)
    return noisy_latents


def decode_input_image(b64_or_url):
    """Decode a base64 string or download a URL to a PIL Image."""
    if b64_or_url.startswith("http://") or b64_or_url.startswith("https://"):
        import requests
        print(f"[TGND] Downloading input image: {b64_or_url[:80]}...", flush=True)
        resp = requests.get(b64_or_url, timeout=60)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    else:
        # Assume base64
        return Image.open(io.BytesIO(base64.b64decode(b64_or_url))).convert("RGB")


def load_yolo_models():
    """Load YOLO face and hand detection models for ADetailer."""
    global yolo_face, yolo_hand
    if yolo_face is not None:
        return

    try:
        from ultralytics import YOLO
        from huggingface_hub import hf_hub_download

        # Check volume cache first
        yolo_dir = "/runpod-volume/yolo" if os.path.exists("/runpod-volume") else "/tmp/yolo"
        os.makedirs(yolo_dir, exist_ok=True)

        face_path = os.path.join(yolo_dir, "face_yolov9c.pt")
        hand_path = os.path.join(yolo_dir, "hand_yolov9c.pt")

        if not os.path.exists(face_path):
            face_path = hf_hub_download("Bingsu/adetailer", "face_yolov9c.pt", local_dir=yolo_dir)
        if not os.path.exists(hand_path):
            hand_path = hf_hub_download("Bingsu/adetailer", "hand_yolov9c.pt", local_dir=yolo_dir)

        yolo_face = YOLO(face_path)
        yolo_hand = YOLO(hand_path)
        print("[TGND] YOLO face+hand models loaded", flush=True)
    except Exception as e:
        print(f"[TGND] Could not load YOLO models: {e}", flush=True)
        yolo_face = False
        yolo_hand = False


def create_feathered_mask(image_size, bbox, feather=20):
    """Create a feathered mask from a bounding box [x1, y1, x2, y2]."""
    w, h = image_size
    mask = Image.new("L", (w, h), 0)

    x1, y1, x2, y2 = [int(v) for v in bbox]

    # Expand bbox by 15% for context
    bw, bh = x2 - x1, y2 - y1
    expand = 0.15
    x1 = max(0, int(x1 - bw * expand))
    y1 = max(0, int(y1 - bh * expand))
    x2 = min(w, int(x2 + bw * expand))
    y2 = min(h, int(y2 + bh * expand))

    # Draw white rectangle
    from PIL import ImageDraw
    draw = ImageDraw.Draw(mask)
    draw.rectangle([x1, y1, x2, y2], fill=255)

    # Feather the edges
    mask = mask.filter(ImageFilter.GaussianBlur(feather))

    return mask


def run_adetailer(image, prompt, seed):
    """Run ADetailer: detect faces/hands, inpaint fixes."""
    load_yolo_models()
    if yolo_face is False:
        return image, {"skipped": "yolo_load_failed"}

    load_inpaint_pipe()
    if inpaint_pipe is False:
        return image, {"skipped": "inpaint_pipe_failed"}

    t0 = time.time()
    img_array = np.array(image)
    stats = {"faces": 0, "hands": 0, "fixed": 0}

    # Detect faces
    face_results = yolo_face(img_array, conf=0.3, verbose=False)
    face_boxes = face_results[0].boxes.xyxy.cpu().numpy() if len(face_results[0].boxes) > 0 else []
    stats["faces"] = len(face_boxes)

    # Detect hands
    hand_results = yolo_hand(img_array, conf=0.3, verbose=False)
    hand_boxes = hand_results[0].boxes.xyxy.cpu().numpy() if len(hand_results[0].boxes) > 0 else []
    stats["hands"] = len(hand_boxes)

    all_boxes = list(face_boxes) + list(hand_boxes)
    if not all_boxes:
        print("[TGND] ADetailer: no faces/hands detected, skipping", flush=True)
        stats["skipped"] = "no_detections"
        return image, stats

    # Inpaint each detected region
    generator = torch.Generator("cuda").manual_seed(seed)

    for i, bbox in enumerate(all_boxes):
        region_type = "face" if i < len(face_boxes) else "hand"
        print(f"[TGND] ADetailer: fixing {region_type} region {bbox}", flush=True)

        mask = create_feathered_mask(image.size, bbox)

        # Inpaint with low strength to fix artifacts while keeping consistency
        result = inpaint_pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,
            strength=0.35,
            guidance_scale=4.0,
            num_inference_steps=25,
            generator=generator,
        )
        image = result.images[0]
        stats["fixed"] += 1

    elapsed = time.time() - t0
    print(f"[TGND] ADetailer done in {elapsed:.1f}s: {stats}", flush=True)
    return image, stats


def download_lora(url_or_path):
    """Download LoRA file, using local path, network volume, or /tmp cache."""
    # Check if it's a local volume path first
    if url_or_path.startswith("/runpod-volume/"):
        if os.path.exists(url_or_path):
            print(f"[TGND] LoRA from local path: {url_or_path}", flush=True)
            return url_or_path
        else:
            print(f"[TGND] WARNING: Local LoRA path not found: {url_or_path}", flush=True)

    url = url_or_path
    url_hash = hashlib.md5(url.encode()).hexdigest()[:12]

    # Check network volume cache first (persists across cold starts)
    volume_cache = f"/runpod-volume/loras/lora_{url_hash}.safetensors"
    if os.path.exists(volume_cache):
        print(f"[TGND] LoRA from volume: {volume_cache}", flush=True)
        return volume_cache

    # Check /tmp cache (persists within same worker)
    tmp_cache = f"/tmp/lora_{url_hash}.safetensors"
    if os.path.exists(tmp_cache):
        print(f"[TGND] LoRA from tmp: {tmp_cache}", flush=True)
        return tmp_cache

    print(f"[TGND] Downloading LoRA: {url[:80]}...", flush=True)
    import requests
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    data = resp.content
    print(f"[TGND] Downloaded {len(data) / 1024 / 1024:.1f}MB", flush=True)

    # Save to network volume if available, otherwise /tmp
    if os.path.exists("/runpod-volume"):
        os.makedirs("/runpod-volume/loras", exist_ok=True)
        with open(volume_cache, "wb") as f:
            f.write(data)
        print(f"[TGND] LoRA saved to volume: {volume_cache}", flush=True)
        return volume_cache
    else:
        with open(tmp_cache, "wb") as f:
            f.write(data)
        return tmp_cache


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
        print(f"[TGND] LoRAs already loaded ({len(lora_configs)} adapters)", flush=True)
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
        print(f"[TGND] Loaded adapter '{name}' (scale={cfg['scale']})", flush=True)

    # Set active adapters with weights
    if len(adapter_names) > 1:
        pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)

    loaded_lora_key = cache_key
    print(f"[TGND] {len(lora_configs)} LoRAs loaded in {time.time() - t0:.1f}s", flush=True)

    # Return scale for joint_attention_kwargs (use max scale)
    return {name: weight for name, weight in zip(adapter_names, adapter_weights)}


def handler(job):
    """RunPod serverless handler."""
    try:
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

        # Generation params (Flux 2 defaults: guidance=4.0, steps=50)
        width = int(inp.get("width", 768))
        height = int(inp.get("height", 1024))
        guidance_scale = float(inp.get("guidance_scale", 4.0))
        num_steps = int(inp.get("num_inference_steps", 50))
        seed = int(inp.get("seed", random.randint(1, 2147483647)))
        use_adetailer = bool(inp.get("adetailer", False))

        # img2img params
        input_image_data = inp.get("image", "")
        strength = float(inp.get("strength", 0.65))

        is_img2img = bool(input_image_data)
        mode = "img2img" if is_img2img else "txt2img"
        print(f"[TGND] {mode}: {width}x{height}, steps={num_steps}, guidance={guidance_scale}, seed={seed}, strength={strength if is_img2img else 'N/A'}, adetailer={use_adetailer}", flush=True)
        t0 = time.time()

        generator = torch.Generator("cuda").manual_seed(seed)

        # Use scale from first adapter if single LoRA
        lora_scale = lora_configs[0]["scale"] if len(lora_configs) == 1 else 1.0
        attn_kwargs = {"scale": lora_scale} if lora_configs else None

        if is_img2img:
            # ─── img2img mode via flow matching latent injection ───
            input_image = decode_input_image(input_image_data)
            input_image = input_image.resize((width, height), Image.LANCZOS)
            print(f"[TGND] Input image decoded and resized to {width}x{height}", flush=True)

            # Prepare noisy latents from input image
            noisy_latents = prepare_img2img_latents(
                input_image, strength, num_steps, width, height, generator
            )

            # Generate from noisy latents — pipeline handles scheduler/mu internally
            result = pipe(
                prompt=prompt,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=num_steps,
                generator=generator,
                latents=noisy_latents,
                attention_kwargs=attn_kwargs,
            )
        else:
            # ─── txt2img mode ───
            result = pipe(
                prompt=prompt,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=num_steps,
                generator=generator,
                attention_kwargs=attn_kwargs,
            )

        image = result.images[0]
        gen_elapsed = time.time() - t0
        print(f"[TGND] Generated in {gen_elapsed:.1f}s ({mode})", flush=True)

        # ADetailer post-processing
        adetailer_stats = None
        if use_adetailer:
            image, adetailer_stats = run_adetailer(image, prompt, seed)

        # Encode to JPEG base64
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=92)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        elapsed = time.time() - t0
        print(f"[TGND] Total time {elapsed:.1f}s, size={len(b64) // 1024}KB", flush=True)

        response = {
            "status": "ok",
            "image": b64,
            "seed": seed,
            "mode": mode,
            "inference_time": round(elapsed, 2),
        }
        if is_img2img:
            response["strength"] = strength
        if adetailer_stats:
            response["adetailer"] = adetailer_stats

        return response

    except Exception as e:
        print(f"[TGND] ERROR in handler: {traceback.format_exc()}", flush=True)
        return {"status": "error", "error": str(e)}


# ─── RunPod entry point ───
print("[TGND] Starting RunPod serverless worker...", flush=True)
runpod.serverless.start({"handler": handler})
