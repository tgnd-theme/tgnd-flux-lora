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
    """Load SegFormer B2 clothes/body-parts model (~400MB VRAM).

    mattmdjaga/segformer_b2_clothes — 18 classes:
      0:Background 1:Hat 2:Hair 3:Sunglasses 4:Upper-clothes 5:Skirt
      6:Pants 7:Dress 8:Belt 9:Left-shoe 10:Right-shoe 11:Face
      12:Left-leg 13:Right-leg 14:Left-arm 15:Right-arm 16:Bag 17:Scarf
    """
    global segformer_model, segformer_processor
    if segformer_model is not None:
        return

    from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
    segformer_processor = SegformerImageProcessor.from_pretrained(
        "mattmdjaga/segformer_b2_clothes",
        cache_dir=MODEL_CACHE,
    )
    segformer_model = AutoModelForSemanticSegmentation.from_pretrained(
        "mattmdjaga/segformer_b2_clothes",
        cache_dir=MODEL_CACHE,
    ).to("cuda").eval()
    log("SegFormer B2 (clothes/body-parts) loaded")


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

def is_bfl_format(filepath):
    """Check if a LoRA file uses BFL/ComfyUI key naming (not diffusers)."""
    from safetensors import safe_open
    with safe_open(filepath, framework="pt") as f:
        keys = f.keys()
        return any("diffusion_model." in k for k in keys)


def convert_bfl_to_diffusers(filepath):
    """Convert BFL/ComfyUI format LoRA to diffusers format using built-in converter."""
    converted_path = filepath.replace(".safetensors", "_diffusers.safetensors")
    if os.path.exists(converted_path):
        log(f"  Using cached converted LoRA: {converted_path}")
        return converted_path

    log(f"  Converting BFL format LoRA to diffusers format...")
    t0 = time.time()
    try:
        from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_lora_to_diffusers
        from safetensors.torch import load_file, save_file

        state_dict = load_file(filepath)
        converted = _convert_non_diffusers_lora_to_diffusers(
            "FluxTransformer2DModel", state_dict
        )
        save_file(converted, converted_path)
        log(f"  Converted: {len(state_dict)}→{len(converted)} keys in {time.time()-t0:.1f}s")
        return converted_path
    except ImportError:
        log(f"  WARNING: _convert_non_diffusers_lora_to_diffusers not available")
        return filepath
    except Exception as e:
        log(f"  WARNING: BFL conversion failed: {e}")
        return filepath


def download_lora(url):
    """Download a LoRA file and cache it on the network volume.
    Auto-converts BFL/ComfyUI format to diffusers format if needed."""
    filename = hashlib.md5(url.encode()).hexdigest() + ".safetensors"
    local_path = os.path.join(LORA_CACHE, filename)

    if not os.path.exists(local_path):
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

    # Auto-convert BFL/ComfyUI format to diffusers
    if is_bfl_format(local_path):
        log(f"  Detected BFL/ComfyUI format LoRA — converting...")
        local_path = convert_bfl_to_diffusers(local_path)

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

        try:
            img2img_pipe.load_lora_weights(local_path, adapter_name=adapter_name)
            # Verify adapter was actually registered
            present = set()
            for comp in [img2img_pipe.transformer, img2img_pipe.text_encoder, img2img_pipe.text_encoder_2]:
                if hasattr(comp, 'peft_config'):
                    present.update(comp.peft_config.keys())
            if adapter_name in present:
                adapter_names.append(adapter_name)
                adapter_weights.append(scale)
                log(f"  LoRA {adapter_name} ({trigger}) loaded OK")
            else:
                log(f"  LoRA {adapter_name} ({trigger}) loaded but not present (incompatible keys?), skipping")
        except Exception as e:
            log(f"  LoRA {adapter_name} ({trigger}) failed: {e}")

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


# SegFormer B2 class ID mappings (mattmdjaga/segformer_b2_clothes)
# Person (all body) = union of skin + clothing classes
SEG_PERSON_IDS = [4, 5, 6, 7, 8, 11, 12, 13, 14, 15]  # clothes + skin
SEG_SKIN_IDS = [11, 12, 13, 14, 15]  # Face, Left-leg, Right-leg, Left-arm, Right-arm


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

    # Body = all person classes (skin + clothing)
    body_mask = get_part_mask(seg_map, SEG_PERSON_IDS, dilate_px=4)

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
    body_mask = get_part_mask(seg_map, SEG_PERSON_IDS, dilate_px=0)

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

_skin_mask_error = None  # Global to report errors in output


def get_skin_mask_hsv(image):
    """Fallback skin detection using HSV color range. Always works, no model needed."""
    import cv2
    arr = np.asarray(image)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)

    # Standard skin color ranges in HSV
    lower1 = np.array([0, 30, 60], dtype=np.uint8)
    upper1 = np.array([20, 180, 255], dtype=np.uint8)
    lower2 = np.array([160, 30, 60], dtype=np.uint8)
    upper2 = np.array([180, 180, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    skin_raw = cv2.bitwise_or(mask1, mask2)

    # Clean up: morphological open (remove noise) then close (fill gaps)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    skin_raw = cv2.morphologyEx(skin_raw, cv2.MORPH_OPEN, kernel)
    skin_raw = cv2.morphologyEx(skin_raw, cv2.MORPH_CLOSE, kernel)

    # Smooth edges
    skin_pil = Image.fromarray(skin_raw)
    skin_pil = skin_pil.filter(ImageFilter.GaussianBlur(radius=5))
    skin_mask = np.asarray(skin_pil, dtype=np.float32) / 255.0

    return skin_mask


def get_skin_mask(image):
    """Get skin mask. Tries SegFormer B2 first, falls back to HSV color detection.

    SegFormer B2 classes: Face(11) Left-leg(12) Right-leg(13) Left-arm(14) Right-arm(15)
    HSV fallback: standard skin color ranges, always works.
    """
    global _skin_mask_error

    # --- Method 1: SegFormer B2 (precise, model-based) ---
    try:
        import cv2
        if segformer_model is not None:
            log(f"  [P5] Trying SegFormer skin detection...")
            seg_map = segment_body(image)
            unique_classes = np.unique(seg_map).tolist()
            log(f"  [P5] SegFormer classes: {unique_classes}")
            skin_binary = np.isin(seg_map, SEG_SKIN_IDS).astype(np.float32)
            skin_pct_raw = skin_binary.mean() * 100

            if skin_pct_raw > 0.5:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
                skin_binary = cv2.erode(skin_binary, kernel, iterations=1)
                skin_pil = Image.fromarray((skin_binary * 255).astype("uint8"))
                skin_pil = skin_pil.filter(ImageFilter.GaussianBlur(radius=5))
                skin_mask = np.asarray(skin_pil, dtype=np.float32) / 255.0
                skin_pct = skin_mask.mean() * 100
                if skin_pct > 0.1:
                    log(f"  [P5] SegFormer skin mask: {skin_pct:.1f}% coverage")
                    _skin_mask_error = None
                    return skin_mask
            log(f"  [P5] SegFormer: only {skin_pct_raw:.1f}% skin, falling back to HSV")
        else:
            log(f"  [P5] SegFormer not loaded, falling back to HSV")
    except Exception as e:
        log(f"  [P5] SegFormer failed ({e}), falling back to HSV")

    # --- Method 2: HSV color detection (fallback, always works) ---
    try:
        skin_mask = get_skin_mask_hsv(image)
        skin_pct = skin_mask.mean() * 100
        if skin_pct > 0.5:
            log(f"  [P5] HSV skin mask: {skin_pct:.1f}% coverage")
            _skin_mask_error = f"hsv_fallback ({skin_pct:.1f}%)"
            return skin_mask
        else:
            log(f"  [P5] HSV: only {skin_pct:.1f}% skin, no texture applied")
            _skin_mask_error = f"hsv_too_low ({skin_pct:.1f}%)"
            return None
    except Exception as e:
        _skin_mask_error = f"both_failed: {e}"
        log(f"  [P5] HSV fallback also failed: {e}")
        return None


def fg_skin_texture(arr, skin_mask, rng):
    """4-layer skin realism pipeline (v12b golden standard). Applies ONLY to skin.

    Layers:
      1. Sub-surface scattering (SSS) — warm color bleeding
      2. Frequency separation retexture — fills missing pore/wrinkle bands
      3. Specular highlight breaking — roughens AI-perfect highlights
      4. Blood perfusion — micro color variation
    """
    import cv2

    if skin_mask is None or skin_mask.max() < 0.01:
        return arr

    h, w = arr.shape[:2]
    img = arr.copy()
    sm = skin_mask

    # --- Layer 1: SSS simulation ---
    sss_r = cv2.GaussianBlur(img[..., 0].astype(np.float32), (0, 0), sigmaX=8)
    sss_g = cv2.GaussianBlur(img[..., 1].astype(np.float32), (0, 0), sigmaX=5)
    sss_b = cv2.GaussianBlur(img[..., 2].astype(np.float32), (0, 0), sigmaX=3)
    sss = np.stack([sss_r, sss_g, sss_b], axis=-1)
    lum = (img @ np.array([0.299, 0.587, 0.114])).astype(np.float32) / 255.0
    sss_weight = np.exp(-((lum - 0.5) ** 2) / (2 * 0.15 ** 2))
    blend = (0.15 * sss_weight * sm)[..., None]
    img = img * (1 - blend) + sss * blend
    shadow_area = ((lum < 0.35) * sm).astype(np.float32)
    shadow_area = cv2.GaussianBlur(shadow_area, (0, 0), sigmaX=5)
    img[..., 0] += 1.5 * shadow_area
    img[..., 2] -= 0.8 * shadow_area

    # --- Layer 2: Frequency separation retexture ---
    gray = (img @ np.array([0.299, 0.587, 0.114])).astype(np.float32)
    low_3 = cv2.GaussianBlur(gray, (0, 0), sigmaX=3)
    existing_hf = gray - low_3
    skin_px = existing_hf[sm > 0.5]
    existing_energy = float(np.std(skin_px)) if len(skin_px) > 100 else 0
    deficit = max(0, 5.0 - existing_energy)
    if deficit > 0.5:
        pore_raw = rng.normal(0, 1, (h, w)).astype(np.float32)
        pore_bp = cv2.GaussianBlur(pore_raw, (0, 0), sigmaX=1.2)
        pore_bp = pore_bp - cv2.GaussianBlur(pore_bp, (0, 0), sigmaX=3.0)
        mid_raw = rng.normal(0, 1, (h, w)).astype(np.float32)
        mid_bp = cv2.GaussianBlur(mid_raw, (0, 0), sigmaX=3.0)
        mid_bp = mid_bp - cv2.GaussianBlur(mid_bp, (0, 0), sigmaX=8.0)
        synth = pore_bp * 0.6 + mid_bp * 0.4
        synth_std = float(np.std(synth))
        if synth_std > 0:
            synth = synth * (deficit / synth_std) * 0.35
        tex_rgb = np.stack([synth * 1.05, synth * 0.95, synth * 0.88], axis=-1)
        midtone = np.clip(1.0 - np.abs(lum - 0.47) / 0.4, 0.3, 1.0)
        img = img + tex_rgb * (sm * midtone)[..., None]

    # --- Layer 3: Specular highlight breaking ---
    brightness = gray / 255.0
    hl_mask = ((brightness > 0.75) * sm).astype(np.float32)
    hl_mask = cv2.GaussianBlur(hl_mask, (0, 0), sigmaX=3)
    if hl_mask.max() > 0.01:
        disrupt = rng.normal(0, 1, (h, w)).astype(np.float32)
        disrupt = cv2.GaussianBlur(disrupt, (0, 0), sigmaX=1.0)
        disrupt = disrupt - cv2.GaussianBlur(disrupt, (0, 0), sigmaX=3.5)
        intensity = np.clip((brightness - 0.75) / 0.25, 0, 1)
        img = img - disrupt * 8.0 * hl_mask * intensity

    # --- Layer 4: Blood perfusion ---
    perf_grid = rng.normal(0, 1, ((h + 49) // 50, (w + 49) // 50)).astype(np.float32)
    perfusion = cv2.resize(perf_grid, (w, h), interpolation=cv2.INTER_CUBIC)
    perf2_grid = rng.normal(0, 1, ((h + 19) // 20, (w + 19) // 20)).astype(np.float32)
    perfusion2 = cv2.resize(perf2_grid, (w, h), interpolation=cv2.INTER_CUBIC)
    color_shift = (perfusion * 0.6 + perfusion2 * 0.4) * 1.5
    img[..., 0] += color_shift * sm * 1.1
    img[..., 1] -= color_shift * sm * 0.3

    return np.clip(img, 0, 255)


def pass5_filmgrade(image, seed):
    """Apply filmgrade + skin texture + anti-AI processing (v12b golden standard).
    Returns (image, skin_texture_applied)."""
    skin_applied = False
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import filmgrade as fg

        p = fg.PRESETS["ouatih"]
        p = dict(p, grain_sigma=3.5)  # v12b golden standard

        rng = np.random.RandomState(seed % (2**31))
        arr = np.asarray(image, dtype=float)

        # Step 1: Filmgrade (color, warmth, grain)
        arr = fg.grade(arr, p)
        arr = fg.halation(Image.fromarray(arr.astype("uint8")), p["halation"])
        arr = fg.add_grain(arr, p["grain_sigma"], rng)
        arr = fg.vignette(arr, p["vignette"])

        # Step 2: 4-layer skin texture (SSS + pores + specular + perfusion)
        skin_mask = get_skin_mask(image)
        if skin_mask is not None:
            arr = fg_skin_texture(arr, skin_mask, rng)
            skin_applied = True
            log(f"  [P5] Skin texture applied (4 layers)")
        else:
            log(f"  [P5] No skin detected, skipping texture")

        # Step 3: Anti-AI processing
        arr = fg.deai(arr, rng, fg.DEAI)

        out = Image.fromarray(arr.astype("uint8"))
        out = fg.jpeg_compress(out, fg.DEAI["jpeg_quality"])

        log(f"  [P5] Filmgrade + anti-AI applied (grain_sigma=3.5, skin={skin_applied})")
        return out, skin_applied
    except Exception as e:
        log(f"  [P5] Filmgrade failed: {e}")
        return image, False


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

        # ── Pass 5: Filmgrade + Skin Texture ──
        skin_texture_applied = False
        if not no_filmgrade:
            image, skin_texture_applied = pass5_filmgrade(image, seed)
            passes_run.append("filmgrade")
            if skin_texture_applied:
                passes_run.append("skin_texture")

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
            "handler_version": "v2.4-hsv-fallback",
            "skin_debug": _skin_mask_error,
        }

    except Exception as e:
        log(f"ERROR: {traceback.format_exc()}")
        return {"status": "error", "error": str(e)}


# ─── RunPod entry point ───
log("Starting RunPod serverless worker (Ultimate Multi-Pass Generator)...")
runpod.serverless.start({"handler": handler})
