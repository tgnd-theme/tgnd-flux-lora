"""
RunPod Serverless Handler — Flux 2 Dev + Multi-LoRA + PuLID + ADetailer inference.

No safety checker. No NSFW filter. Full control.
Triple Stack: LoRA (body/style) + PuLID (face identity) + optional ADetailer (fix artifacts).
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
yolo_person = None
pulid_model = None  # PuLID-Flux2 model (IDFormer + cross-attention modules)


def load_model():
    """Load Flux 2 Dev pipeline once in bfloat16 (no quantization — PuLID needs full weights).
    ~55GB total VRAM on 80GB GPU with PuLID stack."""
    global pipe
    if pipe is not None:
        return

    print("[TGND] Loading Flux 2 Dev pipeline (bfloat16, no quantization)...", flush=True)
    t0 = time.time()

    from diffusers import Flux2Pipeline

    model_id = "black-forest-labs/FLUX.2-dev"
    print(f"[TGND] Using Flux2Pipeline (Flux 2 Dev), model={model_id}", flush=True)

    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)

    # Load in bfloat16 without quantization — PuLID monkey-patches transformer blocks
    # dynamically, which conflicts with quantized weights. ~38GB transformer + ~8GB text encoder.
    pipe = Flux2Pipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
    )

    print(f"[TGND] Pipeline loaded in {time.time() - t0:.1f}s", flush=True)


def load_pulid():
    """Load PuLID-Flux2 model (IDFormer + cross-attention modules)."""
    global pulid_model
    if pulid_model is not None:
        return

    from pulid_flux2 import load_pulid_model, load_face_models

    weights_dir = "/runpod-volume/pulid" if os.path.exists("/runpod-volume") else "/tmp/pulid"
    os.makedirs(weights_dir, exist_ok=True)

    # Download weights from HuggingFace if not cached
    weights_file = os.path.join(weights_dir, "pulid_flux2_klein_v2.safetensors")
    if not os.path.exists(weights_file):
        print("[TGND] Downloading PuLID-Flux2 Klein v2 weights from HuggingFace...", flush=True)
        from huggingface_hub import hf_hub_download
        weights_file = hf_hub_download(
            "Fayens/Pulid-Flux2",
            "pulid_flux2_klein_v2.safetensors",
            local_dir=weights_dir,
            local_dir_use_symlinks=False,
        )
        print(f"[TGND] PuLID weights downloaded to {weights_file}", flush=True)

    pulid_model = load_pulid_model(weights_file, device="cuda")

    # Pre-load face models (InsightFace + EVA-CLIP) to avoid cold-start delay on first request
    load_face_models(device="cuda")
    print("[TGND] PuLID stack fully loaded", flush=True)


def load_inpaint_pipe():
    """Create inpaint pipeline from the main pipeline (shares weights, no extra VRAM)."""
    global inpaint_pipe
    if inpaint_pipe is not None:
        return

    try:
        from diffusers import FluxInpaintPipeline
        inpaint_pipe = FluxInpaintPipeline.from_pipe(pipe)
        print("[TGND] Inpaint pipeline created from main pipe (shared weights)", flush=True)
    except Exception as e:
        print(f"[TGND] Could not create inpaint pipeline: {e}", flush=True)
        inpaint_pipe = False  # sentinel: tried and failed


def load_img2img_pipe():
    """Create img2img pipeline from the main pipeline (shares weights, no extra VRAM).
    Tries FluxImg2ImgPipeline first, falls back to None (use Flux2Pipeline native image param)."""
    global img2img_pipe
    if img2img_pipe is not None:
        return

    try:
        from diffusers import FluxImg2ImgPipeline
        img2img_pipe = FluxImg2ImgPipeline.from_pipe(pipe)
        print("[TGND] Img2img pipeline created from main pipe (shared weights)", flush=True)
    except Exception as e:
        print(f"[TGND] FluxImg2ImgPipeline failed: {e}", flush=True)
        print("[TGND] Will use Flux2Pipeline native image parameter instead", flush=True)
        img2img_pipe = "native"  # sentinel: use Flux2Pipeline.image param


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


def load_yolo_person():
    """Load YOLO segmentation model for person detection."""
    global yolo_person
    if yolo_person is not None:
        return

    try:
        from ultralytics import YOLO

        yolo_dir = "/runpod-volume/yolo" if os.path.exists("/runpod-volume") else "/tmp/yolo"
        os.makedirs(yolo_dir, exist_ok=True)

        model_path = os.path.join(yolo_dir, "yolov8m-seg.pt")
        if not os.path.exists(model_path):
            print("[TGND] Downloading YOLOv8m-seg for person segmentation...", flush=True)
            yolo_person = YOLO("yolov8m-seg.pt")
            # Cache to volume for next cold start
            import shutil
            cached = str(yolo_person.ckpt_path) if hasattr(yolo_person, 'ckpt_path') else None
            if cached and os.path.exists(cached):
                shutil.copy2(cached, model_path)
                print(f"[TGND] YOLOv8m-seg cached to {model_path}", flush=True)
        else:
            yolo_person = YOLO(model_path)

        print("[TGND] YOLOv8m-seg loaded for person segmentation", flush=True)
    except Exception as e:
        print(f"[TGND] Could not load YOLOv8m-seg: {e}", flush=True)
        yolo_person = False


def create_person_mask(image, expand_px=15, feather=35):
    """Detect person in image and create segmentation mask.
    Returns a PIL Image mask (white=replace, black=keep)."""
    load_yolo_person()

    w, h = image.size
    fallback_mask = None

    if yolo_person is False:
        print("[TGND] Person seg unavailable, using center fallback mask", flush=True)
        # Fallback: mask center 70% of image (assumes person is centered)
        fallback_mask = Image.new("L", (w, h), 0)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(fallback_mask)
        margin_x, margin_y = int(w * 0.15), int(h * 0.05)
        draw.ellipse([margin_x, margin_y, w - margin_x, h - margin_y], fill=255)
        return fallback_mask.filter(ImageFilter.GaussianBlur(feather))

    # Detect persons (class 0 in COCO)
    results = yolo_person(np.array(image), classes=[0], conf=0.4, verbose=False)

    if not results or len(results[0].boxes) == 0:
        print("[TGND] No person detected, using center fallback mask", flush=True)
        fallback_mask = Image.new("L", (w, h), 0)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(fallback_mask)
        margin_x, margin_y = int(w * 0.15), int(h * 0.05)
        draw.ellipse([margin_x, margin_y, w - margin_x, h - margin_y], fill=255)
        return fallback_mask.filter(ImageFilter.GaussianBlur(feather))

    # Get segmentation masks — pick the largest person
    if results[0].masks is not None:
        masks_data = results[0].masks.data.cpu().numpy()  # (N, H, W)
        areas = [m.sum() for m in masks_data]
        best_idx = int(np.argmax(areas))
        person_mask = masks_data[best_idx]  # (H, W) float32 0-1

        # Resize mask to image size (YOLO may use different resolution)
        mask_pil = Image.fromarray((person_mask * 255).astype(np.uint8)).resize((w, h), Image.LANCZOS)
    else:
        # No segmentation mask available, use bounding box
        print("[TGND] No seg mask, using bbox", flush=True)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        best_idx = int(np.argmax(areas))
        bbox = boxes[best_idx]
        mask_pil = Image.new("L", (w, h), 0)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(mask_pil)
        draw.rectangle([int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])], fill=255)

    # Expand mask slightly to cover edges
    if expand_px > 0:
        mask_arr = np.array(mask_pil)
        from scipy.ndimage import binary_dilation
        struct = np.ones((expand_px * 2 + 1, expand_px * 2 + 1))
        dilated = binary_dilation(mask_arr > 127, structure=struct)
        mask_pil = Image.fromarray((dilated * 255).astype(np.uint8))

    # Feather edges for smooth blend
    mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(feather))

    person_pct = np.array(mask_pil).mean() / 255 * 100
    print(f"[TGND] Person mask created: {person_pct:.0f}% of image", flush=True)

    return mask_pil


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
        inpaint_person = bool(inp.get("inpaint_person", False))

        # PuLID params
        pulid_image_data = inp.get("pulid_image", "")
        pulid_strength = float(inp.get("pulid_strength", 0.8))

        # DWPose params
        pose_image_data = inp.get("pose_image", "")  # reference image for pose extraction
        validate_pose = bool(inp.get("validate_pose", False))  # compare output pose to input

        # Determine mode
        is_img2img = bool(input_image_data)
        if inpaint_person and is_img2img:
            mode = "inpaint_person"
        elif is_img2img:
            mode = "img2img"
        else:
            mode = "txt2img"

        print(f"[TGND] {mode}: {width}x{height}, steps={num_steps}, guidance={guidance_scale}, seed={seed}, strength={strength if is_img2img else 'N/A'}, pulid={'yes' if pulid_image_data else 'no'}, pose={'yes' if pose_image_data else 'no'}, adetailer={use_adetailer}", flush=True)
        t0 = time.time()

        # ─── DWPose: extract skeleton from reference and enrich prompt ───
        ref_pose_image = None  # keep reference for validation later
        ref_skeleton = None
        if pose_image_data:
            from dwpose_utils import extract_skeleton, keypoints_to_pose_description

            ref_pose_image = decode_input_image(pose_image_data)
            ref_skeleton = extract_skeleton(ref_pose_image, device="cuda")

            if ref_skeleton and ref_skeleton.get("keypoints") is not None:
                pose_desc = keypoints_to_pose_description(ref_skeleton["keypoints"])
                if pose_desc:
                    # Prepend pose description to prompt for better body positioning
                    prompt = f"{pose_desc}, {prompt}"
                    print(f"[TGND] Pose enriched prompt: +'{pose_desc}'", flush=True)
            else:
                print("[TGND] DWPose: no skeleton extracted from reference", flush=True)

        # ─── PuLID face identity injection ───
        unpatch_fn = None
        if pulid_image_data:
            from pulid_flux2 import extract_face_embedding, patch_flux
            load_pulid()

            face_image = decode_input_image(pulid_image_data)
            id_tokens = extract_face_embedding(face_image, device="cuda")

            if id_tokens is not None:
                # Run through IDFormer to get identity tokens
                with torch.no_grad():
                    id_tokens = pulid_model.id_former(id_tokens)  # [1, num_tokens, dim]

                # Monkey-patch transformer blocks with identity cross-attention
                unpatch_fn = patch_flux(
                    pipe.transformer,
                    pulid_model,
                    id_tokens,
                    strength=pulid_strength,
                )
                print(f"[TGND] PuLID active: strength={pulid_strength}", flush=True)
            else:
                print("[TGND] PuLID skipped: no face detected in reference image", flush=True)

        generator = torch.Generator("cuda").manual_seed(seed)

        # Use scale from first adapter if single LoRA
        lora_scale = lora_configs[0]["scale"] if len(lora_configs) == 1 else 1.0
        attn_kwargs = {"scale": lora_scale} if lora_configs else None

        try:
            if mode == "inpaint_person":
                # ─── Person inpainting mode ───
                input_image = decode_input_image(input_image_data)
                input_image = input_image.resize((width, height), Image.LANCZOS)
                print(f"[TGND] Input image decoded and resized to {width}x{height}", flush=True)

                person_mask = create_person_mask(input_image)

                load_inpaint_pipe()
                if inpaint_pipe is False:
                    return {"status": "error", "error": "inpaint pipeline not available"}

                print(f"[TGND] Inpainting person region (strength={strength})", flush=True)
                result = inpaint_pipe(
                    prompt=prompt,
                    image=input_image,
                    mask_image=person_mask,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_steps,
                    generator=generator,
                    attention_kwargs=attn_kwargs,
                )

            elif mode == "img2img":
                # ─── img2img mode ───
                input_image = decode_input_image(input_image_data)
                input_image = input_image.resize((width, height), Image.LANCZOS)
                print(f"[TGND] Input image decoded and resized to {width}x{height}", flush=True)

                load_img2img_pipe()

                if img2img_pipe not in (False, "native"):
                    print(f"[TGND] Using FluxImg2ImgPipeline (strength={strength})", flush=True)
                    result = img2img_pipe(
                        prompt=prompt,
                        image=input_image,
                        strength=strength,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_steps,
                        generator=generator,
                        attention_kwargs=attn_kwargs,
                    )
                else:
                    print(f"[TGND] Using Flux2Pipeline native image param (reference conditioning)", flush=True)
                    result = pipe(
                        prompt=prompt,
                        image=input_image,
                        width=width,
                        height=height,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_steps,
                        generator=generator,
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
        finally:
            # ALWAYS unpatch transformer blocks to prevent stacking across requests
            if unpatch_fn is not None:
                unpatch_fn()

        # ADetailer post-processing
        adetailer_stats = None
        if use_adetailer:
            image, adetailer_stats = run_adetailer(image, prompt, seed)

        # ─── DWPose validation: compare output pose to reference ───
        pose_validation = None
        if validate_pose and ref_pose_image is not None:
            from dwpose_utils import validate_pose as run_pose_validation
            pose_validation = run_pose_validation(ref_pose_image, image, device="cuda")

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
        if mode in ("img2img", "inpaint_person"):
            response["strength"] = strength
        if pulid_image_data:
            response["pulid"] = {"active": unpatch_fn is not None, "strength": pulid_strength}
        if pose_validation:
            response["pose_validation"] = pose_validation
        if adetailer_stats:
            response["adetailer"] = adetailer_stats

        return response

    except Exception as e:
        print(f"[TGND] ERROR in handler: {traceback.format_exc()}", flush=True)
        return {"status": "error", "error": str(e)}


# ─── RunPod entry point ───
print("[TGND] Starting RunPod serverless worker...", flush=True)
runpod.serverless.start({"handler": handler})
