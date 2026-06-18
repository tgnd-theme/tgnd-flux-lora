"""
RunPod Serverless Handler — Ultimate Multi-Pass Lookbook Generation.

Per-escort usage: escort's face+body+style LoRAs are loaded per-request.
Reference photos come from the lookbook (not hardcoded).

Pipeline (same as generate_ultimate.py v12b):
  Pass 1: Base generation (FluxControlNetImg2ImgPipeline + Depth/Pose ControlNet + triple LoRA)
  Pass 2: Hand fix (MediaPipe landmarks + Alimama Inpainting ControlNet on FLUX.1-dev)
  Pass 3: Feet/body fix (SegFormer segmentation + Alimama Inpainting)
  Pass 3e: Chest/nipple fix for topless photos
  Pass 3d: Face restoration (GFPGAN v1.4)
  Pass 5: Filmgrade + anti-AI + skin texture post-processing

Input schema:
{
    "lookbook_image_url": str,     # Reference photo URL (from lookbook)
    "prompt": str,                  # Scene description
    "clothing_prompt": str,         # Clothing prompt snippet
    "clothing_desc": str,           # Clothing detail for fix passes
    "expression_hint": str,         # Expression hint
    "loras": [
        {"url": "...", "scale": 0.5, "trigger": "escort_model"},
        {"url": "...", "scale": 0.9, "trigger": "escort_body"},
        {"url": "...", "scale": 0.5, "trigger": "zishy_style"}
    ],
    "body_description": str,        # Optional full body desc
    "body_overrides": {             # Optional overrides
        "cup": "B", "butt": "round", "build": "slim", "height": "average"
    },
    "strength": float,              # default: 0.80
    "guidance": float,              # default: 5.0
    "seed": int,
    "skip_fix": bool,
    "no_filmgrade": bool,
}
"""

import os
import io
import gc
import sys
import base64
import hashlib
import random
import time
import traceback

try:
    import runpod
    print(f"[TGND-ULT] runpod {runpod.__version__}", flush=True)
except Exception as e:
    print(f"[TGND-ULT] FATAL: cannot import runpod: {e}", flush=True)
    sys.exit(1)

try:
    import torch
    import numpy as np
    from PIL import Image, ImageFilter, ImageDraw
    print(f"[TGND-ULT] torch {torch.__version__}, CUDA: {torch.cuda.is_available()}", flush=True)
except Exception as e:
    print(f"[TGND-ULT] FATAL: cannot import torch/numpy/PIL: {e}", flush=True)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HF_TOKEN = os.environ.get("HF_TOKEN", "")
ANTHROPIC_API_KEY = os.environ.get("TGND_ANTHROPIC_API_KEY", "")

VOLUME_DIR = "/runpod-volume"
LORA_CACHE = os.path.join(VOLUME_DIR, "loras")
MODEL_CACHE = os.path.join(VOLUME_DIR, "models")

# Default generation params (v12b golden standard)
DEFAULT_STRENGTH = 0.80
DEFAULT_GUIDANCE = 5.0
DEFAULT_CN_SCALE = 0.4
DEFAULT_CN_END = 0.5
DEFAULT_POSE_SCALE = 0.3
DEFAULT_POSE_END = 0.4

# Body override mappings
CUP_MAP = {
    "A": "small natural A-cup breasts",
    "B": "natural B-cup breasts",
    "C": "full C-cup breasts",
    "D": "large D-cup breasts",
    "DD+": "very large DD+ breasts",
}
BUTT_MAP = {
    "petite": "petite small butt",
    "athletic": "athletic toned butt",
    "round": "round shapely butt",
    "full": "full voluptuous butt",
}
BUILD_MAP = {
    "slim": "slim slender body",
    "athletic": "athletic toned body",
    "curvy": "curvy hourglass figure",
    "plus": "plus size full figure",
}
HEIGHT_MAP = {
    "petite": "petite short woman",
    "average": "average height woman",
    "tall": "tall statuesque woman",
}

# ---------------------------------------------------------------------------
# Global model state (loaded once on cold start / FlashBoot restore)
# ---------------------------------------------------------------------------
img2img_pipe = None
controlnet = None
single_controlnet = None
segformer_model = None
segformer_processor = None
depth_pipe = None
dwpose_detector = None
mediapipe_hands = None
face_parser_net = None
face_parser_transform = None
gfpgan_restorer = None
loaded_lora_hash = None  # hash of currently loaded LoRA config


def log(msg):
    print(f"[TGND-ULT] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Model Loading (cold start)
# ---------------------------------------------------------------------------

def load_base_models():
    """Load the base img2img pipeline + ControlNet + segmentation models."""
    global img2img_pipe, controlnet, single_controlnet
    if img2img_pipe is not None:
        return

    if HF_TOKEN:
        from huggingface_hub import login
        login(token=HF_TOKEN)

    os.makedirs(LORA_CACHE, exist_ok=True)
    os.makedirs(MODEL_CACHE, exist_ok=True)

    log("Loading FluxControlNetImg2ImgPipeline + Dual ControlNet...")
    t0 = time.time()

    from diffusers import FluxControlNetImg2ImgPipeline, FluxControlNetModel, FluxMultiControlNetModel

    single_controlnet = FluxControlNetModel.from_pretrained(
        "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0",
        torch_dtype=torch.bfloat16,
        cache_dir=MODEL_CACHE,
    ).to("cuda")

    controlnet = FluxMultiControlNetModel([single_controlnet, single_controlnet])

    img2img_pipe = FluxControlNetImg2ImgPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        controlnet=controlnet,
        torch_dtype=torch.bfloat16,
        cache_dir=MODEL_CACHE,
    )
    img2img_pipe.transformer.to("cuda")
    img2img_pipe.text_encoder.to("cuda")
    img2img_pipe.text_encoder_2.to("cuda")
    img2img_pipe.vae.to("cuda")

    elapsed = time.time() - t0
    vram = torch.cuda.max_memory_allocated() / 1e9
    log(f"Pipeline loaded in {elapsed:.0f}s, VRAM: {vram:.1f}GB")

    # Load helper models
    load_segformer()
    load_depth_model()
    load_dwpose()


def load_segformer():
    """Load SegFormer B5 for body segmentation."""
    global segformer_model, segformer_processor
    if segformer_model is not None:
        return

    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    segformer_processor = SegformerImageProcessor.from_pretrained(
        "nvidia/segformer-b5-finetuned-ade-640-640",
        cache_dir=MODEL_CACHE,
    )
    segformer_model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b5-finetuned-ade-640-640",
        cache_dir=MODEL_CACHE,
    ).to("cuda")
    log("SegFormer B5 loaded")


def load_depth_model():
    """Load Depth-Anything V2 for depth map extraction."""
    global depth_pipe
    if depth_pipe is not None:
        return

    from transformers import pipeline
    depth_pipe = pipeline(
        "depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device="cuda",
        model_kwargs={"cache_dir": MODEL_CACHE},
    )
    log("Depth-Anything V2 loaded")


def load_dwpose():
    """Load DWPose detector for pose extraction."""
    global dwpose_detector
    if dwpose_detector is not None:
        return

    try:
        from controlnet_aux import DWposeDetector
        dwpose_cache = os.path.join(MODEL_CACHE, "dwpose")
        dwpose_detector = DWposeDetector.from_pretrained("yzd-v/DWPose", cache_dir=dwpose_cache)
        log("DWPose detector loaded")
    except Exception as e:
        log(f"DWPose not available: {e}")
        dwpose_detector = False


def load_fix_models():
    """Load models needed for fix passes (MediaPipe, face parser, GFPGAN)."""
    global mediapipe_hands, face_parser_net, face_parser_transform, gfpgan_restorer

    if mediapipe_hands is None:
        try:
            import mediapipe as mp
            mediapipe_hands = mp.solutions.hands.Hands(
                static_image_mode=True,
                max_num_hands=2,
                min_detection_confidence=0.3,
            )
            log("MediaPipe Hands loaded")
        except Exception as e:
            log(f"MediaPipe not available: {e}")
            mediapipe_hands = False

    if gfpgan_restorer is None:
        try:
            from gfpgan import GFPGANer
            import cv2

            gfpgan_path = os.path.join(MODEL_CACHE, "GFPGANv1.4.pth")
            if not os.path.exists(gfpgan_path):
                from huggingface_hub import hf_hub_download
                gfpgan_path = hf_hub_download(
                    "TencentARC/GFPGAN",
                    "GFPGANv1.4.pth",
                    local_dir=MODEL_CACHE,
                )

            gfpgan_restorer = GFPGANer(
                model_path=gfpgan_path,
                upscale=1,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=None,
            )
            log("GFPGAN v1.4 loaded")
        except Exception as e:
            log(f"GFPGAN not available: {e}")
            gfpgan_restorer = False


# ---------------------------------------------------------------------------
# LoRA Loading (per-request, cached)
# ---------------------------------------------------------------------------

def download_lora(url):
    """Download a LoRA file and cache it on the network volume."""
    filename = hashlib.md5(url.encode()).hexdigest() + ".safetensors"
    local_path = os.path.join(LORA_CACHE, filename)

    if os.path.exists(local_path):
        return local_path

    import requests
    log(f"Downloading LoRA: {url[:80]}...")
    t0 = time.time()
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()

    os.makedirs(LORA_CACHE, exist_ok=True)
    with open(local_path, "wb") as f:
        f.write(resp.content)

    size_mb = len(resp.content) / 1024 / 1024
    log(f"LoRA cached: {local_path} ({size_mb:.1f}MB) in {time.time()-t0:.1f}s")
    return local_path


def load_loras_for_escort(lora_configs):
    """Load escort-specific LoRAs into the pipeline. Skips reload if same config."""
    global loaded_lora_hash

    if not lora_configs:
        return

    # Hash the config to avoid reloading same LoRAs
    config_hash = hashlib.md5(str(sorted(str(c) for c in lora_configs)).encode()).hexdigest()
    if config_hash == loaded_lora_hash:
        log("LoRAs already loaded (cache hit)")
        return

    # Unload previous LoRAs
    try:
        img2img_pipe.unload_lora_weights()
    except Exception:
        pass

    t0 = time.time()
    adapter_names = []
    adapter_weights = []

    for i, config in enumerate(lora_configs):
        url = config.get("url", "")
        scale = float(config.get("scale", 1.0))
        trigger = config.get("trigger", f"lora_{i}")

        if not url:
            continue

        local_path = download_lora(url)
        adapter_name = f"adapter_{i}"

        img2img_pipe.load_lora_weights(local_path, adapter_name=adapter_name)
        adapter_names.append(adapter_name)
        adapter_weights.append(scale)

    if adapter_names:
        img2img_pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)
        loaded_lora_hash = config_hash
        labels = ", ".join(f"{n}={w}" for n, w in zip(adapter_names, adapter_weights))
        log(f"LoRAs loaded (unfused) [{labels}] in {time.time()-t0:.1f}s")


# ---------------------------------------------------------------------------
# Image Utilities
# ---------------------------------------------------------------------------

def download_image(url):
    """Download an image from URL."""
    import requests
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def extract_depth_map(image, target_size=(768, 1024)):
    """Extract depth map using Depth-Anything V2."""
    img_resized = image.resize(target_size, Image.LANCZOS)
    result = depth_pipe(img_resized)
    depth = result["depth"]
    return depth.resize(target_size, Image.LANCZOS).convert("RGB")


def extract_pose(image, target_size=(768, 1024)):
    """Extract DWPose skeleton."""
    if dwpose_detector is None or dwpose_detector is False:
        return None

    img_resized = image.resize(target_size, Image.LANCZOS)
    skeleton = dwpose_detector(img_resized)
    return skeleton.resize(target_size, Image.LANCZOS)


def segment_body(image):
    """Run SegFormer segmentation, return seg_map numpy array."""
    import torch.nn.functional as F

    inputs = segformer_processor(images=image, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = segformer_model(**inputs)

    logits = outputs.logits
    upsampled = F.interpolate(logits, size=image.size[::-1], mode="bilinear", align_corners=False)
    seg_map = upsampled.argmax(dim=1).squeeze().cpu().numpy()
    return seg_map


def get_part_mask(seg_map, class_ids, dilate_px=8):
    """Create binary mask from segmentation class IDs."""
    import cv2
    mask = np.zeros_like(seg_map, dtype=np.uint8)
    for cid in class_ids:
        mask[seg_map == cid] = 255

    if dilate_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2, dilate_px * 2))
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def get_skin_mask(image):
    """Get skin mask using SegFormer for filmgrade skin texture."""
    seg_map = segment_body(image)
    # ADE20K person class = 12
    skin_mask = get_part_mask(seg_map, [12], dilate_px=0)
    return skin_mask


# ---------------------------------------------------------------------------
# Build Prompt
# ---------------------------------------------------------------------------

def build_body_description(lora_configs, body_overrides=None):
    """Build body description from LoRA triggers + optional overrides."""
    triggers = []
    for config in lora_configs:
        trigger = config.get("trigger", "")
        if trigger:
            triggers.append(trigger)

    # Base body parts
    parts = list(triggers)

    if body_overrides:
        cup = body_overrides.get("cup", "")
        if cup and cup in CUP_MAP:
            parts.append(CUP_MAP[cup])
        butt = body_overrides.get("butt", "")
        if butt and butt in BUTT_MAP:
            parts.append(BUTT_MAP[butt])
        build = body_overrides.get("build", "")
        if build and build in BUILD_MAP:
            parts.append(BUILD_MAP[build])
        height = body_overrides.get("height", "")
        if height and height in HEIGHT_MAP:
            parts.append(HEIGHT_MAP[height])

    parts.append("two arms, two legs, correct human anatomy")
    return ", ".join(parts)


def build_full_prompt(body_desc, expression_hint, scene_prompt, clothing_prompt=""):
    """Combine body, expression, scene, and clothing into full generation prompt."""
    parts = [body_desc]

    if expression_hint:
        parts.append(expression_hint)

    if clothing_prompt:
        parts.append(clothing_prompt)

    if scene_prompt:
        parts.append(scene_prompt)

    return ", ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Pass 1: Base Generation
# ---------------------------------------------------------------------------

def pass1_generate(ref_image, prompt, strength, seed, guidance=5.0,
                   width=768, height=1024):
    """Generate base image with img2img + dual ControlNet + LoRAs."""
    t0 = time.time()

    ref_resized = ref_image.resize((width, height), Image.LANCZOS)

    # Extract depth + pose
    log("  [P1] Extracting depth map...")
    depth_map = extract_depth_map(ref_image, (width, height))

    log("  [P1] Extracting DWPose skeleton...")
    pose_map = extract_pose(ref_image, (width, height))

    generator = torch.Generator("cuda").manual_seed(seed)

    if pose_map is not None:
        control_images = [depth_map, pose_map]
        cn_scales = [DEFAULT_CN_SCALE, DEFAULT_POSE_SCALE]
        cn_ends = [DEFAULT_CN_END, DEFAULT_POSE_END]
        cn_modes = [2, 4]  # depth=2, pose=4

        result = img2img_pipe(
            prompt=prompt,
            image=ref_resized,
            control_image=control_images,
            strength=strength,
            controlnet_conditioning_scale=cn_scales,
            control_guidance_end=cn_ends,
            control_mode=cn_modes,
            width=width,
            height=height,
            num_inference_steps=30,
            guidance_scale=guidance,
            generator=generator,
        )
    else:
        # Fallback: depth only
        result = img2img_pipe(
            prompt=prompt,
            image=ref_resized,
            control_image=[depth_map, Image.new("RGB", (width, height), (0, 0, 0))],
            strength=strength,
            controlnet_conditioning_scale=[DEFAULT_CN_SCALE, 0.0],
            control_guidance_end=[DEFAULT_CN_END, 0.0],
            control_mode=[2, 4],
            width=width,
            height=height,
            num_inference_steps=30,
            guidance_scale=guidance,
            generator=generator,
        )

    image = result.images[0]
    log(f"  [P1] Base generated in {time.time()-t0:.1f}s")
    return image


# ---------------------------------------------------------------------------
# Pass 2: Hand Fix (MediaPipe + Alimama Inpainting)
# ---------------------------------------------------------------------------

def pass2_fix_hands(inpaint_pipe, image, body_prompt, seed):
    """Fix hands using MediaPipe detection + inpainting."""
    if mediapipe_hands is None or mediapipe_hands is False:
        log("  [P2] MediaPipe not available, skipping hand fix")
        return image

    import cv2

    img_np = np.array(image)
    results = mediapipe_hands.process(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

    if not results.multi_hand_landmarks:
        log("  [P2] No hands detected, skipping")
        return image

    # Create mask from hand landmarks
    h, w = img_np.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for hand_landmarks in results.multi_hand_landmarks:
        points = []
        for lm in hand_landmarks.landmark:
            px, py = int(lm.x * w), int(lm.y * h)
            points.append((px, py))

        if points:
            pts = np.array(points, np.int32)
            hull = cv2.convexHull(pts)
            cv2.fillConvexPoly(mask, hull, 255)

    # Dilate mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
    mask = cv2.dilate(mask, kernel, iterations=1)

    mask_pil = Image.fromarray(mask)
    hand_prompt = f"{body_prompt}, detailed realistic hands, five fingers on each hand, correct finger count"

    try:
        generator = torch.Generator("cuda").manual_seed(seed + 100)
        result = inpaint_pipe(
            prompt=hand_prompt,
            image=image,
            mask_image=mask_pil,
            control_image=image,
            strength=0.55,
            num_inference_steps=25,
            guidance_scale=5.0,
            generator=generator,
        )
        log(f"  [P2] Hands fixed ({len(results.multi_hand_landmarks)} hands)")
        return result.images[0]
    except Exception as e:
        log(f"  [P2] Hand fix failed: {e}")
        return image


# ---------------------------------------------------------------------------
# Pass 3: Body Fix (SegFormer + Inpainting)
# ---------------------------------------------------------------------------

def pass3_fix_body(inpaint_pipe, image, body_prompt, clothing_desc, seed):
    """Fix body artifacts using SegFormer segmentation + inpainting."""
    import cv2

    seg_map = segment_body(image)

    # ADE20K: person=12
    body_mask = get_part_mask(seg_map, [12], dilate_px=4)

    # Only fix if body is detected
    body_pixels = np.sum(body_mask > 0)
    if body_pixels < 5000:
        log("  [P3] Body region too small, skipping")
        return image

    # Create foot region mask (bottom 25% of body mask)
    h, w = body_mask.shape
    foot_region = np.zeros_like(body_mask)
    foot_start = int(h * 0.75)
    foot_region[foot_start:] = body_mask[foot_start:]

    if np.sum(foot_region > 0) < 1000:
        log("  [P3] No feet region to fix, skipping")
        return image

    mask_pil = Image.fromarray(foot_region)
    fix_prompt = f"{body_prompt}, {clothing_desc}, realistic feet, correct toes"

    try:
        generator = torch.Generator("cuda").manual_seed(seed + 200)
        result = inpaint_pipe(
            prompt=fix_prompt,
            image=image,
            mask_image=mask_pil,
            control_image=image,
            strength=0.45,
            num_inference_steps=25,
            guidance_scale=5.0,
            generator=generator,
        )
        log(f"  [P3] Feet/body fixed")
        return result.images[0]
    except Exception as e:
        log(f"  [P3] Body fix failed: {e}")
        return image


# ---------------------------------------------------------------------------
# Pass 3e: Chest Fix (topless photos)
# ---------------------------------------------------------------------------

def pass3e_fix_chest(inpaint_pipe, image, body_prompt, seed, clothing_desc=""):
    """Fix chest/nipple artifacts in topless photos."""
    import cv2

    seg_map = segment_body(image)
    body_mask = get_part_mask(seg_map, [12], dilate_px=0)

    h, w = body_mask.shape
    # Chest region: roughly upper-middle body area
    chest_region = np.zeros_like(body_mask)
    chest_top = int(h * 0.25)
    chest_bottom = int(h * 0.55)
    chest_left = int(w * 0.2)
    chest_right = int(w * 0.8)
    chest_region[chest_top:chest_bottom, chest_left:chest_right] = body_mask[chest_top:chest_bottom, chest_left:chest_right]

    if np.sum(chest_region > 0) < 2000:
        log("  [P3e] No chest region detected, skipping")
        return image

    mask_pil = Image.fromarray(chest_region)
    fix_prompt = f"{body_prompt}, natural chest, realistic skin texture, smooth skin"

    try:
        generator = torch.Generator("cuda").manual_seed(seed + 300)
        result = inpaint_pipe(
            prompt=fix_prompt,
            image=image,
            mask_image=mask_pil,
            control_image=image,
            strength=0.35,
            num_inference_steps=20,
            guidance_scale=4.0,
            generator=generator,
        )
        log(f"  [P3e] Chest fixed")
        return result.images[0]
    except Exception as e:
        log(f"  [P3e] Chest fix failed: {e}")
        return image


# ---------------------------------------------------------------------------
# Pass 3d: Face Restoration (GFPGAN)
# ---------------------------------------------------------------------------

def pass3d_restore_face(image, weight=0.7):
    """Restore face using GFPGAN v1.4."""
    if gfpgan_restorer is None or gfpgan_restorer is False:
        log("  [P3d] GFPGAN not available, skipping")
        return image

    import cv2

    try:
        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        _, _, restored_bgr = gfpgan_restorer.enhance(
            img_bgr,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
            weight=weight,
        )

        if restored_bgr is not None:
            restored_rgb = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB)
            log(f"  [P3d] Face restored (weight={weight})")
            return Image.fromarray(restored_rgb)
    except Exception as e:
        log(f"  [P3d] Face restoration failed: {e}")

    return image


# ---------------------------------------------------------------------------
# Pass 5: Filmgrade + Anti-AI + Skin Texture
# ---------------------------------------------------------------------------

def pass5_filmgrade(image, seed):
    """Apply filmgrade (warm Kodak 35mm grade) + anti-AI processing."""
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import filmgrade as fg

        # Use ouatih preset (Once Upon a Time in Hollywood — warm golden Kodak 35mm)
        p = fg.PRESETS["ouatih"]

        # Override grain_sigma to 3.5 (v12b golden standard — lower than default 6.0)
        p = dict(p, grain_sigma=3.5)

        # Create deterministic RNG from seed
        rng = np.random.RandomState(seed % (2**31))
        arr = np.asarray(image, dtype=float)

        # Step 1: Filmgrade (color, warmth, grain)
        arr = fg.grade(arr, p)
        arr = fg.halation(Image.fromarray(arr.astype("uint8")), p["halation"])
        arr = fg.add_grain(arr, p["grain_sigma"], rng)
        arr = fg.vignette(arr, p["vignette"])

        # Step 2: Anti-AI processing (break perfect AI smoothness)
        arr = fg.deai(arr, rng, fg.DEAI)

        out = Image.fromarray(arr.astype("uint8"))

        # Step 3: JPEG compression (real photos always have JPEG artifacts)
        out = fg.jpeg_compress(out, fg.DEAI["jpeg_quality"])

        log(f"  [P5] Filmgrade + anti-AI applied (grain_sigma=3.5)")
        return out
    except Exception as e:
        log(f"  [P5] Filmgrade failed: {e}")
        return image


# ---------------------------------------------------------------------------
# Inpainting Pipeline Management
# ---------------------------------------------------------------------------

def create_inpaint_pipe():
    """Create Alimama inpainting pipeline (swaps ControlNet, reuses FLUX.1-dev)."""
    from diffusers import FluxControlNetInpaintPipeline, FluxControlNetModel

    log("Loading Alimama Inpainting ControlNet...")
    t0 = time.time()

    inpaint_cn = FluxControlNetModel.from_pretrained(
        "alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta",
        torch_dtype=torch.bfloat16,
        cache_dir=MODEL_CACHE,
    ).to("cuda")

    inpaint_pipe = FluxControlNetInpaintPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        controlnet=inpaint_cn,
        torch_dtype=torch.bfloat16,
        cache_dir=MODEL_CACHE,
    )
    inpaint_pipe.transformer.to("cuda")
    inpaint_pipe.text_encoder.to("cuda")
    inpaint_pipe.text_encoder_2.to("cuda")
    inpaint_pipe.vae.to("cuda")

    log(f"Inpaint pipe loaded in {time.time()-t0:.0f}s")
    return inpaint_pipe


# ---------------------------------------------------------------------------
# Remove Zishy Watermark
# ---------------------------------------------------------------------------

def remove_watermark(image, crop_px=40):
    """Remove watermark by cropping bottom pixels and resizing back."""
    w, h = image.size
    cropped = image.crop((0, 0, w, h - crop_px))
    return cropped.resize((w, h), Image.LANCZOS)


# ---------------------------------------------------------------------------
# Main Handler
# ---------------------------------------------------------------------------

def handler(job):
    """RunPod serverless handler — Ultimate multi-pass lookbook generation."""
    try:
        inp = job.get("input", {})

        # Health check
        if inp.get("health_check"):
            load_base_models()
            return {"status": "ok", "message": "healthy", "vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}

        lookbook_url = inp.get("lookbook_image_url", "")
        if not lookbook_url:
            return {"status": "error", "error": "lookbook_image_url is required"}

        lora_configs = inp.get("loras", [])
        if not lora_configs:
            return {"status": "error", "error": "At least one LoRA is required"}

        # Load base models
        load_base_models()

        # Load escort's LoRAs
        load_loras_for_escort(lora_configs)

        # Parse params
        prompt = inp.get("prompt", "")
        clothing_prompt = inp.get("clothing_prompt", "")
        clothing_desc = inp.get("clothing_desc", "")
        expression_hint = inp.get("expression_hint", "")
        body_overrides = inp.get("body_overrides", {})
        body_description = inp.get("body_description", "")
        strength = float(inp.get("strength", DEFAULT_STRENGTH))
        guidance = float(inp.get("guidance", DEFAULT_GUIDANCE))
        seed = int(inp.get("seed", random.randint(1, 2147483647)))
        skip_fix = bool(inp.get("skip_fix", False))
        no_filmgrade = bool(inp.get("no_filmgrade", False))

        # Build body description
        if not body_description:
            body_description = build_body_description(lora_configs, body_overrides)

        # Build full prompt
        full_prompt = build_full_prompt(body_description, expression_hint, prompt, clothing_prompt)

        log(f"Starting generation: seed={seed}, strength={strength}, guidance={guidance}")
        log(f"  Prompt: {full_prompt[:120]}...")
        log(f"  Fix passes: {'OFF' if skip_fix else 'ON'}, Filmgrade: {'OFF' if no_filmgrade else 'ON'}")

        total_t0 = time.time()
        passes_run = []

        # Download reference image
        ref_image = download_image(lookbook_url)
        ref_image = remove_watermark(ref_image)

        # ── Pass 1: Base Generation ──
        image = pass1_generate(ref_image, full_prompt, strength, seed, guidance)
        passes_run.append("base")

        # ── Fix Passes (2, 3, 3e, 3d) ──
        if not skip_fix:
            # Need to swap to inpaint pipeline
            # Free img2img ControlNet VRAM
            log("Swapping to inpaint pipeline for fix passes...")

            # Temporarily unload main ControlNet to free VRAM
            # (We keep the transformer/encoders shared via from_pretrained cache)
            inpaint_pipe_obj = create_inpaint_pipe()

            # Reload LoRAs into inpaint pipe (some may fail due to architecture mismatch)
            inpaint_adapter_names = []
            inpaint_adapter_weights = []
            for i, config in enumerate(lora_configs):
                url = config.get("url", "")
                if not url:
                    continue
                try:
                    local_path = download_lora(url)
                    inpaint_pipe_obj.load_lora_weights(local_path, adapter_name=f"adapter_{i}")
                    # Verify adapter was actually loaded
                    present = set()
                    for comp in [inpaint_pipe_obj.transformer, inpaint_pipe_obj.text_encoder, inpaint_pipe_obj.text_encoder_2]:
                        if hasattr(comp, 'peft_config'):
                            present.update(comp.peft_config.keys())
                    if f"adapter_{i}" in present:
                        inpaint_adapter_names.append(f"adapter_{i}")
                        inpaint_adapter_weights.append(float(config.get("scale", 1.0)))
                        log(f"  Inpaint LoRA adapter_{i} loaded OK")
                    else:
                        log(f"  Inpaint LoRA adapter_{i} loaded but not present (incompatible keys?), skipping")
                except Exception as e:
                    log(f"  Inpaint LoRA adapter_{i} failed (skipping): {e}")

            if inpaint_adapter_names:
                inpaint_pipe_obj.set_adapters(inpaint_adapter_names, adapter_weights=inpaint_adapter_weights)
                log(f"  Inpaint adapters set: {inpaint_adapter_names}")
            else:
                log("  WARNING: No LoRAs loaded into inpaint pipe")

            load_fix_models()

            # Pass 2: Hand fix
            image = pass2_fix_hands(inpaint_pipe_obj, image, body_description, seed)
            passes_run.append("hands")

            # Pass 3: Feet/body fix
            image = pass3_fix_body(inpaint_pipe_obj, image, body_description, clothing_desc, seed)
            passes_run.append("feet")

            # Pass 3e: Chest fix (for topless photos)
            if "topless" in full_prompt.lower() or "nude" in full_prompt.lower():
                image = pass3e_fix_chest(inpaint_pipe_obj, image, body_description, seed, clothing_desc)
                passes_run.append("chest")

            # Pass 3d: Face restoration
            image = pass3d_restore_face(image)
            passes_run.append("face_restore")

            # Cleanup inpaint pipe
            del inpaint_pipe_obj
            gc.collect()
            torch.cuda.empty_cache()

        # ── Pass 5: Filmgrade ──
        if not no_filmgrade:
            image = pass5_filmgrade(image, seed)
            passes_run.append("filmgrade")

        # Encode output
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=92)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        total_elapsed = time.time() - total_t0
        log(f"Done! {len(passes_run)} passes in {total_elapsed:.1f}s: {', '.join(passes_run)}")

        return {
            "status": "ok",
            "image": b64,
            "seed": seed,
            "inference_time": round(total_elapsed, 2),
            "passes_run": passes_run,
        }

    except Exception as e:
        log(f"ERROR: {traceback.format_exc()}")
        return {"status": "error", "error": str(e)}


# ─── RunPod entry point ───
log("Starting RunPod serverless worker (Ultimate Multi-Pass Generator)...")
runpod.serverless.start({"handler": handler})
