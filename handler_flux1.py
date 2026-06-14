"""
RunPod Serverless Handler — Flux 1 Dev + ControlNet Union Pro 2.0 + DWPose + Multi-LoRA + ADetailer.

No safety checker. No NSFW filter. Full control.
No NF4 quantization — bfloat16 only to avoid ControlNet+LoRA bugs.
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
    print(f"[TGND-F1] runpod {runpod.__version__}", flush=True)
except Exception as e:
    print(f"[TGND-F1] FATAL: cannot import runpod: {e}", flush=True)
    sys.exit(1)

try:
    import torch
    print(f"[TGND-F1] torch {torch.__version__}, CUDA: {torch.cuda.is_available()}", flush=True)
except Exception as e:
    print(f"[TGND-F1] FATAL: cannot import torch: {e}", flush=True)
    sys.exit(1)

try:
    import numpy as np
    from PIL import Image, ImageFilter
    print(f"[TGND-F1] numpy {np.__version__}, PIL OK", flush=True)
except Exception as e:
    print(f"[TGND-F1] FATAL: cannot import numpy/PIL: {e}", flush=True)
    sys.exit(1)

# ─── Globals (loaded once on cold start) ───
pipe = None           # FluxControlNetImg2ImgPipeline
controlnet = None     # FluxControlNetModel (Union Pro 2.0)
dwpose = None         # DWPose detector
inpaint_pipe = None
loaded_lora_key = None
yolo_face = None
yolo_hand = None


def load_model():
    """Load Flux 1 Dev + ControlNet Union Pro 2.0 pipeline (bfloat16, NO quantization)."""
    global pipe, controlnet
    if pipe is not None:
        return

    print("[TGND-F1] Loading Flux 1 Dev + ControlNet pipeline...", flush=True)
    t0 = time.time()

    from diffusers import FluxControlNetImg2ImgPipeline, FluxControlNetModel

    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)

    # Load ControlNet Union Pro 2.0 (DWPose, canny, depth — all in one model)
    print("[TGND-F1] Loading Shakker-Labs ControlNet Union Pro 2.0...", flush=True)
    controlnet = FluxControlNetModel.from_pretrained(
        "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0",
        torch_dtype=torch.bfloat16,
    )

    # Load Flux 1 Dev with ControlNet — NO NF4 to avoid LoRA+ControlNet bugs
    # bfloat16: ~24GB transformer + ~10GB T5-XXL + ~1GB CLIP + ~1GB VAE + ~4GB ControlNet = ~40GB
    print("[TGND-F1] Loading FLUX.1-dev pipeline (bfloat16, no quantization)...", flush=True)
    pipe = FluxControlNetImg2ImgPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        controlnet=controlnet,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
    )

    elapsed = time.time() - t0
    print(f"[TGND-F1] Pipeline loaded in {elapsed:.1f}s", flush=True)

    # Log VRAM usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[TGND-F1] VRAM: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved", flush=True)


def extract_dwpose(image):
    """Extract DWPose skeleton from a PIL Image for ControlNet conditioning."""
    global dwpose
    t0 = time.time()

    if dwpose is None:
        print("[TGND-F1] Loading DWPose detector...", flush=True)
        from controlnet_aux import DWposeDetector
        dwpose = DWposeDetector.from_pretrained("yzd-v/DWPose", cache_dir="/tmp/dwpose")
        print("[TGND-F1] DWPose detector loaded", flush=True)

    skeleton = dwpose(image)
    elapsed = time.time() - t0
    print(f"[TGND-F1] DWPose extracted in {elapsed:.1f}s", flush=True)
    return skeleton


def load_inpaint_pipe():
    """Create inpaint pipeline from the main pipeline (shares weights, no extra VRAM)."""
    global inpaint_pipe
    if inpaint_pipe is not None:
        return

    try:
        from diffusers import FluxInpaintPipeline
        inpaint_pipe = FluxInpaintPipeline.from_pipe(pipe)
        print("[TGND-F1] Inpaint pipeline created from main pipe (shared weights)", flush=True)
    except Exception as e:
        print(f"[TGND-F1] Could not create inpaint pipeline: {e}", flush=True)
        inpaint_pipe = False  # sentinel: tried and failed


def decode_input_image(b64_or_url):
    """Decode a base64 string or download a URL to a PIL Image."""
    if b64_or_url.startswith("http://") or b64_or_url.startswith("https://"):
        import requests
        print(f"[TGND-F1] Downloading input image: {b64_or_url[:80]}...", flush=True)
        resp = requests.get(b64_or_url, timeout=60)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    else:
        return Image.open(io.BytesIO(base64.b64decode(b64_or_url))).convert("RGB")


def load_yolo_models():
    """Load YOLO face and hand detection models for ADetailer."""
    global yolo_face, yolo_hand
    if yolo_face is not None:
        return

    try:
        from ultralytics import YOLO
        from huggingface_hub import hf_hub_download

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
        print("[TGND-F1] YOLO face+hand models loaded", flush=True)
    except Exception as e:
        print(f"[TGND-F1] Could not load YOLO models: {e}", flush=True)
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

    from PIL import ImageDraw
    draw = ImageDraw.Draw(mask)
    draw.rectangle([x1, y1, x2, y2], fill=255)
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
        print("[TGND-F1] ADetailer: no faces/hands detected, skipping", flush=True)
        stats["skipped"] = "no_detections"
        return image, stats

    generator = torch.Generator("cuda").manual_seed(seed)

    for i, bbox in enumerate(all_boxes):
        region_type = "face" if i < len(face_boxes) else "hand"
        print(f"[TGND-F1] ADetailer: fixing {region_type} region {bbox}", flush=True)

        mask = create_feathered_mask(image.size, bbox)

        result = inpaint_pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,
            strength=0.35,
            guidance_scale=3.5,
            num_inference_steps=25,
            generator=generator,
        )
        image = result.images[0]
        stats["fixed"] += 1

    elapsed = time.time() - t0
    print(f"[TGND-F1] ADetailer done in {elapsed:.1f}s: {stats}", flush=True)
    return image, stats


def download_lora(url_or_path):
    """Download LoRA file, using local path, network volume, or /tmp cache."""
    if url_or_path.startswith("/runpod-volume/"):
        if os.path.exists(url_or_path):
            print(f"[TGND-F1] LoRA from local path: {url_or_path}", flush=True)
            return url_or_path
        else:
            print(f"[TGND-F1] WARNING: Local LoRA path not found: {url_or_path}", flush=True)

    url = url_or_path
    url_hash = hashlib.md5(url.encode()).hexdigest()[:12]

    # Check network volume cache first
    volume_cache = f"/runpod-volume/loras/lora_{url_hash}.safetensors"
    if os.path.exists(volume_cache):
        print(f"[TGND-F1] LoRA from volume: {volume_cache}", flush=True)
        return volume_cache

    # Check /tmp cache
    tmp_cache = f"/tmp/lora_{url_hash}.safetensors"
    if os.path.exists(tmp_cache):
        print(f"[TGND-F1] LoRA from tmp: {tmp_cache}", flush=True)
        return tmp_cache

    print(f"[TGND-F1] Downloading LoRA: {url[:80]}...", flush=True)
    import requests
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    data = resp.content
    print(f"[TGND-F1] Downloaded {len(data) / 1024 / 1024:.1f}MB", flush=True)

    # Save to network volume if available, otherwise /tmp
    if os.path.exists("/runpod-volume"):
        os.makedirs("/runpod-volume/loras", exist_ok=True)
        with open(volume_cache, "wb") as f:
            f.write(data)
        print(f"[TGND-F1] LoRA saved to volume: {volume_cache}", flush=True)
        return volume_cache
    else:
        with open(tmp_cache, "wb") as f:
            f.write(data)
        return tmp_cache


def load_loras(lora_configs):
    """Load one or more LoRAs. Caches the combination."""
    global loaded_lora_key

    if not lora_configs:
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
        print(f"[TGND-F1] LoRAs already loaded ({len(lora_configs)} adapters)", flush=True)
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
        print(f"[TGND-F1] Loaded adapter '{name}' (scale={cfg['scale']})", flush=True)

    # Set active adapters with weights
    if len(adapter_names) > 1:
        pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)

    loaded_lora_key = cache_key
    print(f"[TGND-F1] {len(lora_configs)} LoRAs loaded in {time.time() - t0:.1f}s", flush=True)

    return {name: weight for name, weight in zip(adapter_names, adapter_weights)}


def apply_filmgrade(image, preset, seed):
    """Apply filmgrade post-processing if available."""
    if not preset:
        return image

    try:
        from filmgrade import filmgrade
        image = filmgrade(image, preset=preset, seed=seed)
        print(f"[TGND-F1] Filmgrade applied (preset={preset})", flush=True)
    except Exception as e:
        print(f"[TGND-F1] Filmgrade failed: {e}", flush=True)

    return image


def handler(job):
    """RunPod serverless handler — Flux 1 Dev + ControlNet + LoRA."""
    try:
        inp = job.get("input", {})

        prompt = inp.get("prompt", "")
        if not prompt:
            return {"status": "error", "error": "prompt is required"}

        # image is required — this is an img2img + ControlNet pipeline
        image_data = inp.get("image", "")
        if not image_data:
            return {"status": "error", "error": "image is required (escort photo for img2img basis)"}

        # Load model on first request
        load_model()

        # Build LoRA config list
        lora_configs = inp.get("loras", [])
        if not lora_configs:
            lora_url = inp.get("lora_url", "")
            if lora_url:
                lora_configs = [{"url": lora_url, "scale": float(inp.get("lora_scale", 1.0))}]

        adapters = load_loras(lora_configs)

        # Generation params (Flux 1 defaults: guidance=3.5, steps=30)
        width = int(inp.get("width", 768))
        height = int(inp.get("height", 1024))
        guidance_scale = float(inp.get("guidance_scale", 3.5))
        num_steps = int(inp.get("num_inference_steps", 30))
        seed = int(inp.get("seed", random.randint(1, 2147483647)))
        use_adetailer = bool(inp.get("adetailer", True))
        filmgrade_preset = inp.get("filmgrade", "ouatih")

        # img2img params
        strength = float(inp.get("strength", 0.6))

        # ControlNet params
        controlnet_scale = float(inp.get("controlnet_scale", 0.8))
        control_end = float(inp.get("control_end", 0.65))

        # Pose reference image (optional — for DWPose extraction)
        pose_image_data = inp.get("pose_image", "")

        has_pose = bool(pose_image_data)
        mode = "img2img+controlnet" if has_pose else "img2img"
        print(f"[TGND-F1] {mode}: {width}x{height}, steps={num_steps}, guidance={guidance_scale}, "
              f"seed={seed}, strength={strength}, cn_scale={controlnet_scale}, cn_end={control_end}, "
              f"adetailer={use_adetailer}, filmgrade={filmgrade_preset}", flush=True)

        t0 = time.time()
        generator = torch.Generator("cuda").manual_seed(seed)

        # Decode escort photo (img2img basis)
        input_image = decode_input_image(image_data)
        input_image = input_image.resize((width, height), Image.LANCZOS)
        print(f"[TGND-F1] Input image decoded and resized to {width}x{height}", flush=True)

        # LoRA scale — use joint_attention_kwargs for Flux 1
        lora_scale = lora_configs[0]["scale"] if len(lora_configs) == 1 else 1.0
        joint_attn_kwargs = {"scale": lora_scale} if lora_configs else None

        if has_pose:
            # ─── img2img + ControlNet mode ───
            # Extract DWPose skeleton from pose reference
            pose_ref = decode_input_image(pose_image_data)
            pose_skeleton = extract_dwpose(pose_ref)
            pose_skeleton = pose_skeleton.resize((width, height), Image.LANCZOS)

            result = pipe(
                prompt=prompt,
                image=input_image,
                control_image=pose_skeleton,
                strength=strength,
                controlnet_conditioning_scale=controlnet_scale,
                control_guidance_end=control_end,
                width=width,
                height=height,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                joint_attention_kwargs=joint_attn_kwargs,
            )
        else:
            # ─── img2img only mode (no pose reference) ───
            # Use the pipeline without ControlNet conditioning by passing a blank control image
            # This effectively does img2img with the loaded ControlNet ignored
            blank_control = Image.new("RGB", (width, height), (0, 0, 0))

            result = pipe(
                prompt=prompt,
                image=input_image,
                control_image=blank_control,
                strength=strength,
                controlnet_conditioning_scale=0.0,  # zero scale = ControlNet disabled
                width=width,
                height=height,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                joint_attention_kwargs=joint_attn_kwargs,
            )

        image = result.images[0]
        gen_elapsed = time.time() - t0
        print(f"[TGND-F1] Generated in {gen_elapsed:.1f}s ({mode})", flush=True)

        # ADetailer post-processing
        adetailer_stats = None
        if use_adetailer:
            image, adetailer_stats = run_adetailer(image, prompt, seed)

        # Filmgrade post-processing
        if filmgrade_preset:
            image = apply_filmgrade(image, filmgrade_preset, seed)

        # Encode to JPEG base64
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=92)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        elapsed = time.time() - t0
        print(f"[TGND-F1] Total time {elapsed:.1f}s, size={len(b64) // 1024}KB", flush=True)

        response = {
            "status": "ok",
            "image": b64,
            "seed": seed,
            "mode": mode,
            "strength": strength,
            "inference_time": round(elapsed, 2),
        }
        if has_pose:
            response["controlnet_scale"] = controlnet_scale
            response["control_end"] = control_end
        if adetailer_stats:
            response["adetailer"] = adetailer_stats

        return response

    except Exception as e:
        print(f"[TGND-F1] ERROR in handler: {traceback.format_exc()}", flush=True)
        return {"status": "error", "error": str(e)}


# ─── RunPod entry point ───
print("[TGND-F1] Starting RunPod serverless worker (Flux 1 Dev + ControlNet)...", flush=True)
runpod.serverless.start({"handler": handler})
