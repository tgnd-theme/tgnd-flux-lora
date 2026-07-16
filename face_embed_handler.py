#!/usr/bin/env python3
"""
Cabine42 face-identity service (RunPod serverless worker).

The differentiator (16 jul, founder: "we moeten ons echt onderscheiden … zo goed mogelijk"):
a rigorous, deterministic identity check — not a subjective "does it look like her". Uses
InsightFace/ArcFace 512-D face embeddings (the same family the PuLID path already relies on).

Two jobs, one worker:
  1. build a creator's FACE SIGNATURE from her training photos (robust centroid of per-photo
     embeddings) — computed once at training, stored per creator;
  2. SCORE any image (a calibration test shot, or a live generated photo) against a signature via
     cosine similarity — a precise 0..1 number with calibrated thresholds. Powers the calibration
     gate AND runtime drift-detection ("every photo verified to be really her").

Input (RunPod job "input"):
  { "mode": "signature", "images": [<base64>, ...] }              -> { "signature": [512 floats], "faces": N }
  { "mode": "score", "signature": [512 floats], "image": <b64> }  -> { "score": 0..1, "face_found": bool }
  { "mode": "embed", "images": [<base64>, ...] }                  -> { "embeddings": [[512], ...] }
Images may be raw base64 or data URLs. Largest detected face per image is used.
"""
import base64
import io
import numpy as np
from PIL import Image
import runpod
from insightface.app import FaceAnalysis

# buffalo_l = RetinaFace detector + ArcFace(w600k_r50) recognition — the standard, accurate combo.
_app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
_app.prepare(ctx_id=0, det_size=(640, 640))


def _to_bgr(b64):
    if isinstance(b64, str) and b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    return np.array(img)[:, :, ::-1]  # RGB -> BGR for insightface


def _largest_face_embedding(bgr):
    faces = _app.get(bgr)
    if not faces:
        return None
    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
    emb = faces[0].normed_embedding  # already L2-normalised, 512-D
    return np.asarray(emb, dtype=np.float32)


def _embeddings(images):
    out = []
    for b64 in images or []:
        try:
            e = _largest_face_embedding(_to_bgr(b64))
            if e is not None:
                out.append(e)
        except Exception:
            continue
    return out


def _cosine(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    n = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / n) if n > 0 else 0.0


def handler(job):
    inp = job.get("input", {}) or {}
    mode = inp.get("mode", "score")

    if mode == "signature":
        embs = _embeddings(inp.get("images", []))
        if not embs:
            return {"error": "no faces found in any training photo"}
        # Robust centroid: mean of L2-normalised embeddings, re-normalised.
        centroid = np.mean(np.stack(embs), axis=0)
        centroid = centroid / (np.linalg.norm(centroid) or 1.0)
        return {"signature": centroid.astype(float).tolist(), "faces": len(embs)}

    if mode == "embed":
        return {"embeddings": [e.astype(float).tolist() for e in _embeddings(inp.get("images", []))]}

    # mode == "score": one image vs a stored signature. Cosine mapped to 0..1 (ArcFace cosine is
    # already ~0..1 for same-identity; clamp negatives from different faces to 0).
    sig = inp.get("signature")
    img = inp.get("image")
    if not sig or not img:
        return {"error": "score needs 'signature' and 'image'"}
    e = None
    try:
        e = _largest_face_embedding(_to_bgr(img))
    except Exception as ex:
        return {"error": f"decode/detect failed: {ex}"}
    if e is None:
        return {"score": 0.0, "face_found": False}
    return {"score": max(0.0, _cosine(e, sig)), "face_found": True}


runpod.serverless.start({"handler": handler})
