"""
PuLID-FLUX — Face identity injection for FLUX.1-dev.

Ported from ToTheBeginning/PuLID (guozinan/PuLID weights v0.9.1).

Architecture:
  InsightFace (512D) + EVA-CLIP (768D + 5x hidden features) -> IDFormer -> [B, 32, 2048]
  -> 20 PerceiverAttentionCA modules injected into Flux transformer blocks
  -> every 2nd double block + every 4th single block

FLUX.1-dev: 19 double blocks + 38 single blocks, hidden dim = 3072.
"""

import gc
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# --- FeedForward ---
def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


def reshape_tensor(x, heads):
    bs, length, width = x.shape
    x = x.view(bs, length, heads, -1)
    x = x.transpose(1, 2)
    x = x.reshape(bs, heads, length, -1)
    return x


# --- PerceiverAttentionCA ---
# Cross-attention module injected into Flux transformer blocks.
# Queries from image hidden states (dim=3072), KV from identity tokens (kv_dim=2048).
class PerceiverAttentionCA(nn.Module):
    def __init__(self, *, dim=3072, dim_head=128, heads=16, kv_dim=2048):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm1 = nn.LayerNorm(dim if kv_dim is None else kv_dim)
        self.norm2 = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim if kv_dim is None else kv_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """
        x: identity tokens [B, num_id, kv_dim=2048]
        latents: image hidden states [B, seq, dim=3072]
        Returns: correction [B, seq, dim=3072]
        """
        x = self.norm1(x)
        latents = self.norm2(latents)
        b, seq_len, _ = latents.shape
        q = self.to_q(latents)
        k, v = self.to_kv(x).chunk(2, dim=-1)
        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v
        out = out.permute(0, 2, 1, 3).reshape(b, seq_len, -1)
        return self.to_out(out)


# --- PerceiverAttention (used inside IDFormer) ---
# Differs from CA version: concatenates x + latents for KV.
class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8, kv_dim=None):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm1 = nn.LayerNorm(dim if kv_dim is None else kv_dim)
        self.norm2 = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim if kv_dim is None else kv_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        x = self.norm1(x)
        latents = self.norm2(latents)
        b, seq_len, _ = latents.shape
        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v
        out = out.permute(0, 2, 1, 3).reshape(b, seq_len, -1)
        return self.to_out(out)


# --- IDFormer ---
# Perceiver resampler that converts face embeddings + multi-scale CLIP features
# into identity tokens for injection.
class IDFormer(nn.Module):
    def __init__(self, dim=1024, depth=10, dim_head=64, heads=16, num_id_token=5,
                 num_queries=32, output_dim=2048, ff_mult=4):
        super().__init__()
        self.num_id_token = num_id_token
        self.dim = dim
        self.num_queries = num_queries
        assert depth % 5 == 0
        self.depth = depth // 5  # 2 layers per block
        scale = dim ** -0.5

        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) * scale)
        self.proj_out = nn.Parameter(scale * torch.randn(dim, output_dim))

        self.layers = nn.ModuleList([
            nn.ModuleList([
                PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                FeedForward(dim=dim, mult=ff_mult),
            ]) for _ in range(depth)
        ])

        # 5 mapping MLPs for multi-scale CLIP features
        for i in range(5):
            setattr(self, f'mapping_{i}', nn.Sequential(
                nn.Linear(1024, 1024),
                nn.LayerNorm(1024),
                nn.LeakyReLU(),
                nn.Linear(1024, 1024),
                nn.LayerNorm(1024),
                nn.LeakyReLU(),
                nn.Linear(1024, dim),
            ))

        # Maps ArcFace(512) + CLIP_CLS(768) = 1280 -> num_id_token * dim tokens
        self.id_embedding_mapping = nn.Sequential(
            nn.Linear(1280, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, dim * num_id_token),
        )

    def forward(self, x, y):
        """
        x: [B, 1280] concatenated ArcFace + CLIP CLS embeddings
        y: list of 5 tensors, each [B, seq, 1024] EVA-CLIP hidden features
        Returns: [B, num_queries, output_dim] identity tokens
        """
        latents = self.latents.repeat(x.size(0), 1, 1)

        # Handle multi-face (x might be [B, N, 1280])
        num_duotu = x.shape[1] if x.ndim == 3 else 1

        x = self.id_embedding_mapping(x)
        x = x.reshape(-1, self.num_id_token * num_duotu, self.dim)

        latents = torch.cat((latents, x), dim=1)

        for i in range(5):
            vit_feature = getattr(self, f'mapping_{i}')(y[i])
            ctx_feature = torch.cat((x, vit_feature), dim=1)
            for attn, ff in self.layers[i * self.depth: (i + 1) * self.depth]:
                latents = attn(ctx_feature, latents) + latents
                latents = ff(latents) + latents

        latents = latents[:, :self.num_queries]
        latents = latents @ self.proj_out
        return latents


# --- PuLID-FLUX Model ---
# Holds IDFormer (pulid_encoder) + 20 CA injection modules (pulid_ca).

class PuLIDFlux(nn.Module):
    def __init__(self, device="cuda", weight_dtype=torch.bfloat16):
        super().__init__()
        self.device = device
        self.weight_dtype = weight_dtype

        # Injection intervals (from original PuLID-FLUX)
        self.double_interval = 2  # every 2nd double block
        self.single_interval = 4  # every 4th single block

        # Encoder
        self.pulid_encoder = IDFormer()

        # Calculate number of CA modules
        num_ca = 19 // self.double_interval + 38 // self.single_interval
        if 19 % self.double_interval != 0:
            num_ca += 1
        if 38 % self.single_interval != 0:
            num_ca += 1
        # = 10 + 10 = 20

        self.pulid_ca = nn.ModuleList([
            PerceiverAttentionCA() for _ in range(num_ca)
        ])

    def load_pretrain(self, weights_path):
        """Load PuLID-FLUX weights from safetensors."""
        from safetensors.torch import load_file

        print(f"[PuLID] Loading weights from {weights_path}...", flush=True)
        state_dict = load_file(weights_path, device=str(self.device))

        # Split state_dict by top-level module prefix
        state_dict_split = {}
        for k, v in state_dict.items():
            module = k.split('.')[0]
            state_dict_split.setdefault(module, {})
            new_k = k[len(module) + 1:]
            state_dict_split[module][new_k] = v

        for module_name in state_dict_split:
            print(f"[PuLID] Loading {module_name} ({len(state_dict_split[module_name])} keys)", flush=True)
            getattr(self, module_name).load_state_dict(state_dict_split[module_name], strict=True)

        self.to(self.device, self.weight_dtype)
        self.eval()

        total_params = sum(p.numel() for p in self.parameters()) / 1e6
        print(f"[PuLID] Model loaded: {total_params:.1f}M params, {len(self.pulid_ca)} CA modules", flush=True)

        del state_dict
        del state_dict_split


# --- Face embedding extraction ---

_face_app = None
_face_handler = None
_clip_model = None
_clip_preprocess_mean = None
_clip_preprocess_std = None
_face_parse_model = None
_face_helper = None


def load_face_models(device="cuda"):
    """Load InsightFace + EVA-CLIP + face parser models."""
    global _face_app, _face_handler, _clip_model, _clip_preprocess_mean, _clip_preprocess_std
    global _face_parse_model, _face_helper

    if _face_app is not None:
        return

    import os

    # --- InsightFace ---
    print("[PuLID] Loading InsightFace antelopev2...", flush=True)
    from insightface.app import FaceAnalysis

    cache_dir = "/runpod-volume/insightface" if os.path.exists("/runpod-volume") else "/tmp/insightface"
    os.makedirs(cache_dir, exist_ok=True)

    model_dir = os.path.join(cache_dir, "models", "antelopev2")
    if not os.path.exists(model_dir) or len(os.listdir(model_dir)) < 4:
        print("[PuLID] Downloading antelopev2 models...", flush=True)
        os.makedirs(model_dir, exist_ok=True)
        from huggingface_hub import hf_hub_download
        for model_file in ["1k3d68.onnx", "2d106det.onnx", "genderage.onnx", "glintr100.onnx", "scrfd_10g_bnkps.onnx"]:
            hf_hub_download("DIAMONIK7777/antelopev2", model_file, local_dir=model_dir, local_dir_use_symlinks=False)

    _face_app = FaceAnalysis(
        name="antelopev2", root=cache_dir,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    _face_app.prepare(ctx_id=0, det_size=(640, 640))

    # Also load glintr100 directly for aligned face fallback
    glintr_path = os.path.join(model_dir, "glintr100.onnx")
    if os.path.exists(glintr_path):
        import insightface
        _face_handler = insightface.model_zoo.get_model(
            glintr_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        _face_handler.prepare(ctx_id=0)
    print("[PuLID] InsightFace loaded", flush=True)

    # --- EVA-CLIP ---
    print("[PuLID] Loading EVA-CLIP...", flush=True)
    import open_clip

    clip_cache = "/runpod-volume/clip" if os.path.exists("/runpod-volume") else None
    model, _, preprocess = open_clip.create_model_and_transforms(
        "EVA02-L-14-336", pretrained="merged2b_s6b_b61k", cache_dir=clip_cache,
    )
    _clip_model = model.visual.to(device=device, dtype=torch.bfloat16)
    _clip_model.eval()

    # Get normalization stats
    _clip_preprocess_mean = getattr(_clip_model, 'image_mean', (0.48145466, 0.4578275, 0.40821073))
    _clip_preprocess_std = getattr(_clip_model, 'image_std', (0.26862954, 0.26130258, 0.27577711))
    if not isinstance(_clip_preprocess_mean, (list, tuple)):
        _clip_preprocess_mean = (_clip_preprocess_mean,) * 3
    if not isinstance(_clip_preprocess_std, (list, tuple)):
        _clip_preprocess_std = (_clip_preprocess_std,) * 3
    print("[PuLID] EVA-CLIP loaded", flush=True)

    # --- Face parser (bisenet) ---
    print("[PuLID] Loading face parser...", flush=True)
    try:
        from facexlib.parsing import init_parsing_model
        from facexlib.utils.face_restoration_helper import FaceRestoreHelper

        _face_helper = FaceRestoreHelper(
            upscale_factor=1, face_size=512, crop_ratio=(1, 1),
            det_model='retinaface_resnet50', save_ext='png', device=device,
        )
        _face_helper.face_parse = init_parsing_model(model_name='bisenet', device=device)
        print("[PuLID] Face parser loaded", flush=True)
    except Exception as e:
        print(f"[PuLID] Face parser failed to load: {e}. Will use simple CLIP extraction.", flush=True)
        _face_helper = None


def _to_gray(img):
    """Convert to grayscale while keeping 3 channels."""
    x = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
    return x.repeat(1, 3, 1, 1)


def _get_clip_hidden_features(image_tensor, device="cuda"):
    """
    Run EVA-CLIP visual model and capture 5 evenly-spaced hidden features.
    Uses forward hooks since open_clip doesn't have return_hidden parameter.

    Returns:
        clip_cls: [1, 768] CLS token features
        hidden_features: list of 5 [1, seq, 1024] tensors
    """
    hidden_states = []
    hooks = []

    # EVA02-L-14-336 has 24 transformer layers
    # We capture from layers at ~5 evenly-spaced positions
    # Original PuLID uses indices based on the EVA-CLIP implementation
    transformer = _clip_model.trunk if hasattr(_clip_model, 'trunk') else _clip_model

    # Find the transformer blocks
    if hasattr(transformer, 'blocks'):
        blocks = transformer.blocks
    elif hasattr(transformer, 'transformer') and hasattr(transformer.transformer, 'resblocks'):
        blocks = transformer.transformer.resblocks
    else:
        # Fallback: try to find blocks
        for name, module in transformer.named_modules():
            if 'blocks' in name and isinstance(module, nn.Sequential):
                blocks = module
                break
        else:
            print("[PuLID] WARNING: Could not find transformer blocks for hidden features", flush=True)
            return None, []

    num_blocks = len(blocks)
    # Capture from 5 evenly-spaced layers
    capture_indices = [int(i * (num_blocks - 1) / 4) for i in range(5)]

    def make_hook(idx):
        def hook_fn(module, input, output):
            # output shape: [B, seq, dim]
            if isinstance(output, tuple):
                hidden_states.append(output[0].detach())
            else:
                hidden_states.append(output.detach())
        return hook_fn

    for idx in capture_indices:
        h = blocks[idx].register_forward_hook(make_hook(idx))
        hooks.append(h)

    # Run forward pass
    with torch.no_grad():
        clip_output = _clip_model(image_tensor)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Get CLS token (clip_output is the final pooled output)
    if isinstance(clip_output, tuple):
        clip_cls = clip_output[0]
    else:
        clip_cls = clip_output

    # Ensure clip_cls is [B, 768]
    if clip_cls.ndim == 3:
        clip_cls = clip_cls[:, 0]  # Take CLS token

    return clip_cls, hidden_states


@torch.no_grad()
def extract_face_embedding(image, device="cuda"):
    """
    Extract face identity embedding from a PIL Image.

    Args:
        image: PIL.Image.Image - face reference photo

    Returns:
        id_embedding: [1, 32, 2048] tensor - identity tokens for CA injection
        Or None if no face detected
    """
    import cv2
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.functional import normalize, resize

    load_face_models(device)

    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # --- InsightFace: detect face ---
    face_info = _face_app.get(img_bgr)
    id_ante_embedding = None
    if len(face_info) > 0:
        # Use largest face
        face_info = sorted(face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[-1]
        id_ante_embedding = face_info['embedding']

    if id_ante_embedding is None and _face_handler is None:
        print("[PuLID] WARNING: No face detected", flush=True)
        return None

    # --- Face alignment + parsing ---
    face_features_image = None
    if _face_helper is not None:
        try:
            _face_helper.clean_all()
            _face_helper.read_image(img_bgr)
            _face_helper.get_face_landmarks_5(only_center_face=True)
            _face_helper.align_warp_face()

            if len(_face_helper.cropped_faces) > 0:
                align_face = _face_helper.cropped_faces[0]

                # Fallback: if InsightFace didn't detect, use aligned face
                if id_ante_embedding is None and _face_handler is not None:
                    id_ante_embedding = _face_handler.get_feat(align_face)

                # Parse face to remove background
                from torchvision.transforms.functional import normalize as tv_normalize

                def img2tensor(img, bgr2rgb=True):
                    if bgr2rgb:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    return torch.from_numpy(img.transpose(2, 0, 1)).float()

                input_tensor = img2tensor(align_face, bgr2rgb=True).unsqueeze(0) / 255.0
                input_tensor = input_tensor.to(device)

                parsing_out = _face_helper.face_parse(
                    tv_normalize(input_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                )[0]
                parsing_out = parsing_out.argmax(dim=1, keepdim=True)

                # Background labels to mask out
                bg_label = [0, 16, 18, 7, 8, 9, 14, 15]
                bg = sum(parsing_out == i for i in bg_label).bool()
                white_image = torch.ones_like(input_tensor)
                face_features_image = torch.where(bg, white_image, _to_gray(input_tensor))
        except Exception as e:
            print(f"[PuLID] Face alignment/parsing failed: {e}. Using full image.", flush=True)

    if id_ante_embedding is None:
        print("[PuLID] WARNING: No face detected after all attempts", flush=True)
        return None

    id_ante_embedding = torch.from_numpy(id_ante_embedding).to(device=device, dtype=torch.bfloat16)
    if id_ante_embedding.ndim == 1:
        id_ante_embedding = id_ante_embedding.unsqueeze(0)

    # --- EVA-CLIP: get CLS + hidden features ---
    if face_features_image is None:
        # Fallback: use full image (no face parsing available)
        from torchvision.transforms.functional import to_tensor
        face_features_image = to_tensor(image).unsqueeze(0).to(device)

    # Resize to CLIP input size
    clip_size = getattr(_clip_model, 'image_size', (336, 336))
    if isinstance(clip_size, int):
        clip_size = (clip_size, clip_size)
    face_features_image = resize(face_features_image, list(clip_size), InterpolationMode.BICUBIC)
    face_features_image = normalize(face_features_image, list(_clip_preprocess_mean), list(_clip_preprocess_std))

    id_cond_vit, id_vit_hidden = _get_clip_hidden_features(
        face_features_image.to(dtype=torch.bfloat16), device=device
    )

    if id_cond_vit is None:
        print("[PuLID] WARNING: CLIP feature extraction failed", flush=True)
        return None

    # Normalize CLIP features
    id_cond_vit_norm = torch.norm(id_cond_vit, 2, 1, True)
    id_cond_vit = torch.div(id_cond_vit, id_cond_vit_norm)

    # Concatenate: [1, 512+768] = [1, 1280]
    id_cond = torch.cat([id_ante_embedding, id_cond_vit], dim=-1)

    print(f"[PuLID] Face embedding: ante={id_ante_embedding.shape}, clip={id_cond_vit.shape}, hidden={len(id_vit_hidden)} layers", flush=True)

    return id_cond, id_vit_hidden


# --- Monkey-patching ---

def patch_flux(transformer, pulid_model, id_embedding, strength=0.8):
    """
    Monkey-patch Flux transformer blocks to inject PuLID identity.

    Args:
        transformer: FluxTransformer2DModel
        pulid_model: PuLIDFlux instance with loaded weights
        id_embedding: [B, 32, 2048] from IDFormer (already processed)
        strength: injection strength (0.0 = no effect, 1.0 = full)

    Returns:
        unpatch function
    """
    double_blocks = getattr(transformer, "transformer_blocks", [])
    single_blocks = getattr(transformer, "single_transformer_blocks", [])
    n_double = len(double_blocks)
    n_single = len(single_blocks)

    double_interval = pulid_model.double_interval
    single_interval = pulid_model.single_interval

    print(f"[PuLID] Patching: {n_double} double (every {double_interval}) + {n_single} single (every {single_interval}), strength={strength}", flush=True)

    original_forwards = {}
    ca_idx = 0

    # Patch double blocks (every double_interval-th)
    for idx in range(n_double):
        if idx % double_interval != 0:
            continue

        original_forwards[('double', idx)] = double_blocks[idx].forward
        ca_module = pulid_model.pulid_ca[ca_idx]
        ca_idx += 1

        def make_double_patch(orig_fwd, ca, block_idx):
            def patched_forward(*args, **kwargs):
                result = orig_fwd(*args, **kwargs)

                if isinstance(result, tuple) and len(result) == 2:
                    img, txt = result
                    correction = ca(id_embedding, img)
                    img = img + strength * correction
                    return (img, txt)
                return result
            return patched_forward

        double_blocks[idx].forward = make_double_patch(
            original_forwards[('double', idx)], ca_module, idx
        )

    # Patch single blocks (every single_interval-th)
    for idx in range(n_single):
        if idx % single_interval != 0:
            continue

        original_forwards[('single', idx)] = single_blocks[idx].forward
        ca_module = pulid_model.pulid_ca[ca_idx]
        ca_idx += 1

        def make_single_patch(orig_fwd, ca, block_idx):
            def patched_forward(*args, **kwargs):
                result = orig_fwd(*args, **kwargs)

                if isinstance(result, tuple):
                    out = result[0]
                else:
                    out = result

                correction = ca(id_embedding, out)
                out = out + strength * correction

                if isinstance(result, tuple):
                    return (out,) + result[1:]
                return out
            return patched_forward

        single_blocks[idx].forward = make_single_patch(
            original_forwards[('single', idx)], ca_module, idx
        )

    print(f"[PuLID] Patched {ca_idx} blocks total", flush=True)

    def unpatch():
        for key, orig_fwd in original_forwards.items():
            block_type, block_idx = key
            if block_type == 'double':
                double_blocks[block_idx].forward = orig_fwd
            else:
                single_blocks[block_idx].forward = orig_fwd
        print("[PuLID] Unpatched transformer blocks", flush=True)

    return unpatch


# --- High-level API ---

def load_pulid_model(weights_path, device="cuda"):
    """Load PuLID-FLUX model from safetensors weights."""
    model = PuLIDFlux(device=device)
    model.load_pretrain(weights_path)
    return model
