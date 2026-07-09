"""
Invisible forensic watermark — embeds a compact per-generation ID into every delivered photo.

Used by Cabine42 (consent-based photo studio) to give each generated photo a hidden, traceable
mark, at zero ongoing cost (no third-party subscription — see project memory, decision made 9 jul 2026
after comparing to Privly/CopyrightShark, which charge ~$63/mo for active leak-monitoring we deliberately
do NOT build or offer).

HONESTY NOTE (validated locally 9 jul 2026 before this was wired into the handler — see test results in
Cabine42 project memory):
  - Survives: the exact file being re-shared/re-downloaded, and moderate recompression (JPEG quality
    ~85 and above). Confirmed with real embed/decode/recompress tests.
  - Does NOT reliably survive: resizing (even a 90% resize breaks it), or heavier recompression
    (~JPEG quality 60 and below). This means it will usually NOT survive a screenshot-and-repost, which
    is the most common real-world leak vector for this kind of content.
  - A checksum byte is embedded alongside the ID specifically so a corrupted/absent watermark is reported
    as "no verifiable mark" (None) rather than a wrong attribution. Never claim a match unless the
    checksum passes — a false positive (accusing the wrong session) is worse than no signal at all.
  - Conclusion: this is "best-effort direct-copy protection", not a leak-detection system. Do not market
    it as catching screenshots — user-facing copy should say only that a mark exists and can prove
    origin for the unmodified/lightly-recompressed file.

Method: dwtDctSvd (frequency-domain embedding) via the open-source `invisible-watermark` library
(ShieldMnt/invisible-watermark, MIT-licensed, the same technique Stability AI ships in production for
Stable Diffusion outputs). No GPU required — this is plain CPU image processing.
"""

import hashlib

import cv2
import numpy as np
from PIL import Image

try:
    from imwatermark import WatermarkEncoder, WatermarkDecoder
except Exception as e:  # pragma: no cover - surfaced clearly at call time, not at import time
    WatermarkEncoder = None
    WatermarkDecoder = None
    _IMPORT_ERROR = e

ID_LEN = 6                       # bytes — compact id, derived from a session/attachment id
PAYLOAD_LEN = ID_LEN + 1         # + 1 checksum byte


def _checksum(id_bytes: bytes) -> int:
    return sum(id_bytes) % 256


def make_id_from_string(s: str) -> bytes:
    """Derive a stable 6-byte id from an arbitrary string (e.g. a session/job id)."""
    return hashlib.sha256(s.encode("utf-8")).digest()[:ID_LEN]


def id_to_hex(id_bytes: bytes) -> str:
    return id_bytes.hex()


def embed_watermark(pil_image: Image.Image, id_bytes: bytes) -> Image.Image:
    """Embed a 6-byte id + checksum into a PIL image. Returns a NEW PIL image (RGB)."""
    if WatermarkEncoder is None:
        raise RuntimeError(f"invisible-watermark not available: {_IMPORT_ERROR}")
    if len(id_bytes) != ID_LEN:
        raise ValueError(f"watermark id must be exactly {ID_LEN} bytes, got {len(id_bytes)}")

    payload = id_bytes + bytes([_checksum(id_bytes)])

    rgb = np.array(pil_image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    encoder = WatermarkEncoder()
    encoder.set_watermark("bytes", payload)
    watermarked_bgr = encoder.encode(bgr, "dwtDctSvd")

    watermarked_rgb = cv2.cvtColor(watermarked_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(watermarked_rgb)


def decode_watermark(pil_image: Image.Image):
    """Try to decode + verify a watermark. Returns the 6-byte id bytes on success, else None.

    Returns None both when there is no watermark and when the watermark is corrupted beyond the
    checksum's ability to verify it (e.g. after a resize or heavy recompression) — by design, this
    function never returns a value it can't verify.
    """
    if WatermarkDecoder is None:
        raise RuntimeError(f"invisible-watermark not available: {_IMPORT_ERROR}")

    rgb = np.array(pil_image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    decoder = WatermarkDecoder("bytes", PAYLOAD_LEN * 8)
    try:
        decoded = decoder.decode(bgr, "dwtDctSvd")
    except Exception:
        return None
    if len(decoded) != PAYLOAD_LEN:
        return None
    body, chk = decoded[:ID_LEN], decoded[ID_LEN]
    if _checksum(body) == chk:
        return body
    return None
