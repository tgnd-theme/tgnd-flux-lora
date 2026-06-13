"""
PuLID-Flux2 — Face identity injection for Flux 2 Dev.

Ported from community PuLID-Flux2 implementations:
- iFayens/ComfyUI-PuLID-Flux2
- Fayens/Pulid-Flux2 (HuggingFace weights)

Architecture:
  InsightFace (512D) + EVA-CLIP (768D) → IDFormer (4 perceiver layers) → identity tokens
  → monkey-patch Flux transformer blocks with cross-attention injection
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── PerceiverAttentionCA ───
# Cross-attention module injected into each Flux transformer block.
# Queries = image hidden states, Keys/Values = identity tokens from IDFormer.

class PerceiverAttentionCA(nn.Module):
    def __init__(self, dim=6144, dim_head=128, heads=16, kv_dim=None):
        super().__init__()
        if kv_dim is None:
            kv_dim = dim
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.norm1 = nn.LayerNorm(kv_dim)   # norm for identity tokens (KV)
        self.norm2 = nn.LayerNorm(dim)       # norm for image tokens (Q)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(kv_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """
        x: identity tokens [B, num_id, kv_dim] — from IDFormer
        latents: image hidden states [B, seq, dim] — from Flux transformer block
        Returns: correction to add to latents [B, seq, dim]
        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, seq_len, _ = latents.shape

        q = self.to_q(latents)
        kv = self.to_kv(x)
        k, v = kv.chunk(2, dim=-1)

        # Reshape for multi-head attention
        q = q.view(b, seq_len, self.heads, self.dim_head).transpose(1, 2)
        k = k.view(b, -1, self.heads, self.dim_head).transpose(1, 2)
        v = v.view(b, -1, self.heads, self.dim_head).transpose(1, 2)

        # Scaled dot-product attention
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b, seq_len, -1)

        return self.to_out(out)


# ─── IDFormer ───
# Perceiver-resampler that converts face embeddings to identity tokens.

class IDFormer(nn.Module):
    def __init__(self, dim=6144, num_tokens=4, num_layers=4):
        super().__init__()
        self.num_tokens = num_tokens

        # Project ArcFace (512) + EVA-CLIP CLS (768) = 1280 → dim tokens
        self.proj = nn.Sequential(
            nn.Linear(1280, dim),
            nn.GELU(),
            nn.Linear(dim, dim * num_tokens),
        )

        # Learnable latent queries
        self.latents = nn.Parameter(torch.randn(1, num_tokens, dim) * 0.02)

        # Perceiver cross-attention layers
        self.layers = nn.ModuleList([
            PerceiverAttentionCA(dim=dim, kv_dim=dim) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, id_cond):
        """
        id_cond: [B, 1280] — concatenated ArcFace + EVA-CLIP CLS embeddings
        Returns: [B, num_tokens, dim] — identity tokens for injection
        """
        b = id_cond.shape[0]

        # Project to token sequence
        x = self.proj(id_cond)  # [B, dim * num_tokens]
        x = x.view(b, self.num_tokens, -1)  # [B, num_tokens, dim]

        # Learnable queries
        latents = self.latents.expand(b, -1, -1)  # [B, num_tokens, dim]

        # Cross-attend: latents query into projected face embeddings
        for layer in self.layers:
            latents = latents + layer(x, latents)

        return self.norm(latents)  # [B, num_tokens, dim]


# ─── PuLIDFlux2 ───
# Main module: holds IDFormer + per-block cross-attention modules.

class PuLIDFlux2(nn.Module):
    def __init__(self, dim=6144, num_double_ca=12, num_single_ca=60):
        super().__init__()
        self.dim = dim
        self.id_former = IDFormer(dim=dim)
        self.double_ca = nn.ModuleList([
            PerceiverAttentionCA(dim=dim) for _ in range(num_double_ca)
        ])
        self.single_ca = nn.ModuleList([
            PerceiverAttentionCA(dim=dim) for _ in range(num_single_ca)
        ])

    @classmethod
    def from_pretrained(cls, path, device="cpu"):
        """Load PuLID-Flux2 weights from safetensors file."""
        from safetensors.torch import load_file

        state_dict = load_file(path, device=str(device))

        # Auto-detect dim from latents shape
        latents_key = "id_former.latents"
        if latents_key in state_dict:
            dim = state_dict[latents_key].shape[-1]
        else:
            dim = 6144

        # Count CA modules
        num_double = sum(1 for k in state_dict if k.startswith("double_ca.") and k.endswith(".to_q.weight"))
        num_single = sum(1 for k in state_dict if k.startswith("single_ca.") and k.endswith(".to_q.weight"))

        print(f"[PuLID] Detected: dim={dim}, double_ca={num_double}, single_ca={num_single}", flush=True)

        model = cls(dim=dim, num_double_ca=num_double, num_single_ca=num_single)
        model.load_state_dict(state_dict, strict=True)
        return model


# ─── Flux variant detection ───

def detect_flux_variant(transformer):
    """Detect Flux model variant from transformer block counts."""
    double_blocks = getattr(transformer, "transformer_blocks", [])
    single_blocks = getattr(transformer, "single_transformer_blocks", [])
    n_double = len(double_blocks)
    n_single = len(single_blocks)

    if n_double <= 6 and n_single <= 22:
        return "klein_4b", 3072, n_double, n_single
    elif n_double <= 10 and n_single <= 30:
        return "klein_9b", 4096, n_double, n_single
    else:
        return "flux2_dev", 6144, n_double, n_single


# ─── Scale factors ───
# Progressive scaling: early blocks get stronger identity injection,
# later blocks get weaker (preserves pose/composition).

def get_scale_factor(block_idx, total_blocks, block_type, variant):
    """Get progressive scale factor for a specific block."""
    if total_blocks <= 1:
        return 1.0

    progress = block_idx / (total_blocks - 1)

    if block_type == "double":
        # 8.0 → 3.0 (early blocks strongest for identity)
        return 8.0 - 5.0 * progress
    else:
        # Single blocks: 6.5 → 1.8 (4-zone progressive)
        if progress < 0.25:
            return 6.5
        elif progress < 0.5:
            return 4.5
        elif progress < 0.75:
            return 3.0
        else:
            return 1.8


def get_ca_index(block_idx, total_blocks, num_ca):
    """Map block index to cross-attention module index."""
    if total_blocks <= num_ca:
        return min(block_idx, num_ca - 1)
    return min(int(block_idx * num_ca / total_blocks), num_ca - 1)


# ─── Monkey-patching ───

def patch_flux(transformer, pulid_module, id_tokens, strength=0.8):
    """
    Monkey-patch Flux transformer blocks to inject PuLID identity.

    Args:
        transformer: Flux2Pipeline's transformer (FluxTransformer2DModel)
        pulid_module: PuLIDFlux2 instance with loaded weights
        id_tokens: [B, num_tokens, dim] from IDFormer
        strength: overall injection strength (0.0 = no effect, 1.0 = full)

    Returns:
        unpatch function — MUST be called after generation to restore original forwards
    """
    double_blocks = getattr(transformer, "transformer_blocks", [])
    single_blocks = getattr(transformer, "single_transformer_blocks", [])
    variant, dim, n_double, n_single = detect_flux_variant(transformer)

    print(f"[PuLID] Patching {variant}: {n_double} double + {n_single} single blocks, strength={strength}", flush=True)

    original_double_fwd = {}
    original_single_fwd = {}

    # Patch double blocks
    for idx, block in enumerate(double_blocks):
        original_double_fwd[idx] = block.forward

        def make_double_patch(block_idx, orig_fwd):
            def patched_forward(*args, **kwargs):
                # Call original forward
                result = orig_fwd(*args, **kwargs)

                # result is (hidden_states, encoder_hidden_states) for double blocks
                if isinstance(result, tuple) and len(result) == 2:
                    img, txt = result
                else:
                    return result

                # Apply PuLID cross-attention
                ca_idx = get_ca_index(block_idx, n_double, len(pulid_module.double_ca))
                ca = pulid_module.double_ca[ca_idx]
                factor = get_scale_factor(block_idx, n_double, "double", variant)

                correction = ca(id_tokens, img)
                correction = F.normalize(correction, p=2, dim=-1)
                img = img + strength * factor * correction

                return (img, txt)
            return patched_forward

        block.forward = make_double_patch(idx, original_double_fwd[idx])

    # Patch single blocks
    for idx, block in enumerate(single_blocks):
        original_single_fwd[idx] = block.forward

        def make_single_patch(block_idx, orig_fwd):
            def patched_forward(*args, **kwargs):
                result = orig_fwd(*args, **kwargs)

                # Single blocks return combined tensor (txt + img concatenated)
                # or just hidden_states depending on diffusers version
                if isinstance(result, tuple):
                    out = result[0]
                else:
                    out = result

                ca_idx = get_ca_index(block_idx, n_single, len(pulid_module.single_ca))
                ca = pulid_module.single_ca[ca_idx]
                factor = get_scale_factor(block_idx, n_single, "single", variant)

                correction = ca(id_tokens, out)
                correction = F.normalize(correction, p=2, dim=-1)
                out = out + strength * factor * correction

                if isinstance(result, tuple):
                    return (out,) + result[1:]
                return out
            return patched_forward

        block.forward = make_single_patch(idx, original_single_fwd[idx])

    # Return unpatch function
    def unpatch():
        for idx, block in enumerate(double_blocks):
            if idx in original_double_fwd:
                block.forward = original_double_fwd[idx]
        for idx, block in enumerate(single_blocks):
            if idx in original_single_fwd:
                block.forward = original_single_fwd[idx]
        print("[PuLID] Unpatched transformer blocks", flush=True)

    return unpatch


# ─── Face embedding extraction ───

_face_app = None
_clip_model = None
_clip_preprocess = None


def load_face_models(device="cuda"):
    """Load InsightFace + EVA-CLIP models (cached)."""
    global _face_app, _clip_model, _clip_preprocess

    if _face_app is not None:
        return

    import os
    import numpy as np

    # ─── InsightFace ───
    print("[PuLID] Loading InsightFace antelopev2...", flush=True)
    from insightface.app import FaceAnalysis

    # Cache to network volume
    cache_dir = "/runpod-volume/insightface" if os.path.exists("/runpod-volume") else "/tmp/insightface"
    os.makedirs(cache_dir, exist_ok=True)

    _face_app = FaceAnalysis(
        name="antelopev2",
        root=cache_dir,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    _face_app.prepare(ctx_id=0, det_size=(640, 640))
    print("[PuLID] InsightFace loaded", flush=True)

    # ─── EVA-CLIP ───
    print("[PuLID] Loading EVA-CLIP...", flush=True)
    import open_clip

    clip_cache = "/runpod-volume/clip" if os.path.exists("/runpod-volume") else None
    _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
        "EVA02-CLIP-L-14-336",
        pretrained="merged2s_s6b_b61k",
        cache_dir=clip_cache,
    )
    _clip_model = _clip_model.visual.to(device=device, dtype=torch.bfloat16)
    _clip_model.eval()
    print("[PuLID] EVA-CLIP loaded", flush=True)


def extract_face_embedding(image, device="cuda"):
    """
    Extract face identity embedding from a PIL Image.

    Args:
        image: PIL.Image.Image — face reference photo
        device: torch device

    Returns:
        id_cond: [1, 1280] tensor — concatenated ArcFace + CLIP embeddings
        Or None if no face detected
    """
    import numpy as np

    load_face_models(device)

    # ─── InsightFace: detect face and get 512D embedding ───
    img_array = np.array(image)
    # InsightFace expects BGR
    img_bgr = img_array[:, :, ::-1].copy()

    faces = _face_app.get(img_bgr)
    if not faces:
        print("[PuLID] WARNING: No face detected in reference image", flush=True)
        return None

    # Pick largest face
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    arcface_emb = torch.from_numpy(face.normed_embedding).unsqueeze(0)  # [1, 512]
    arcface_emb = arcface_emb.to(device=device, dtype=torch.bfloat16)

    # ─── EVA-CLIP: extract 768D visual features ───
    clip_input = _clip_preprocess(image).unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        clip_features = _clip_model(clip_input)  # [1, 768]
    clip_features = F.normalize(clip_features, p=2, dim=-1)

    # ─── Concatenate: [1, 512+768] = [1, 1280] ───
    id_cond = torch.cat([arcface_emb, clip_features], dim=-1)

    print(f"[PuLID] Face embedding extracted: arcface={arcface_emb.shape}, clip={clip_features.shape} → {id_cond.shape}", flush=True)
    return id_cond


# ─── High-level API ───

def load_pulid_model(weights_path, device="cuda"):
    """Load PuLID-Flux2 model from safetensors weights."""
    print(f"[PuLID] Loading weights from {weights_path}...", flush=True)
    model = PuLIDFlux2.from_pretrained(weights_path, device=device)
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    print(f"[PuLID] Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params", flush=True)
    return model
