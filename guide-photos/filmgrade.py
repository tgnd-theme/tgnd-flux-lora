#!/usr/bin/env python3
"""
filmgrade.py - Bakt de Once Upon a Time-look in foto's.

Warme, gouden, vivid en SCHONE 35mm-grade (Once Upon a Time in Hollywood).
Deterministisch per bestand (seed = hash van bestandsnaam): dezelfde foto geeft
altijd hetzelfde resultaat, verschillende foto's krijgen net andere fijne korrel.

Dit is puur een foto-grade. Het raakt de site/het design niet.

Gebruik:
    python3 filmgrade.py foto.jpg                  # -> foto_graded.jpg
    python3 filmgrade.py *.jpg --out verwerkt/      # batch
    python3 filmgrade.py foto.jpg --preset zishy    # nog lichtere touch

Presets:
    ouatih (default)  Once Upon a Time in Hollywood: warm, gouden, vivid, SCHOON.
                      De default voor dame-foto's. Matcht Zishy, poetst de
                      echtheid (huidtextuur, tan lines) niet weg.
    zishy             nog lichtere touch, voor een al sterk gouden harde-zon shot
                      waar ouatih te veel wordt.
"""

import os, argparse, hashlib
import numpy as np
from PIL import Image, ImageFilter

# Warme schone foto-grade in vaste getallen. Beide presets zitten in dezelfde
# familie; ouatih is de default, zishy is lichter. Geen schade, geen krassen.
PRESETS = {
    "ouatih": dict(
        # Once Upon a Time in Hollywood: warm, gouden, vivid, SCHOON. Kodak 35mm.
        saturation=1.04,      # vivid kleur, popt (geen ontkleuring)
        black_lift=8,         # schone zwarten, heel licht warm opgetild
        white_drop=250,       # zonovergoten highlights blijven
        warm_r=1.07,
        warm_g=1.03,
        warm_b=0.92,          # gouden warmte, kleur blijft leven
        shadow_warm=(14, 8, 0),
        grain_sigma=6.0,      # heel fijne 35mm-korrel
        halation=0.20,        # zachte zonnegloed / film bloom
        vignette=0.93,        # nauwelijks vignet
    ),
    "zishy": dict(
        # Nog lichtere touch. Behoudt brightness en naturel maximaal.
        saturation=1.00,
        black_lift=6,
        white_drop=251,
        warm_r=1.05,
        warm_g=1.02,
        warm_b=0.95,
        shadow_warm=(10, 6, 0),
        grain_sigma=5.0,
        halation=0.14,
        vignette=0.95,
    ),
}


def _rng(name):
    h = int(hashlib.sha256(name.encode()).hexdigest(), 16) % (2**32)
    return np.random.default_rng(h)


def grade(arr, p):
    """Warme schone 35mm-grade. arr float 0-255."""
    lum = arr @ np.array([0.299, 0.587, 0.114])
    lum = lum[..., None]
    arr = lum + (arr - lum) * p["saturation"]            # saturatie
    arr[..., 0] *= p["warm_r"]                           # warme white balance
    arr[..., 1] *= p["warm_g"]
    arr[..., 2] *= p["warm_b"]
    arr = (arr - 128) * 1.02 + 128                       # licht rijk contrast
    arr = p["black_lift"] + arr * (p["white_drop"] - p["black_lift"]) / 255.0
    shadow_mask = np.clip(1.0 - lum / 110.0, 0, 1)       # warme schaduwen
    arr += shadow_mask * np.array(p["shadow_warm"], dtype=float)
    return np.clip(arr, 0, 255)


def halation(img, strength):
    """Warme zonnegloed die uit de highlights bloedt (film bloom)."""
    a = np.asarray(img, dtype=float)
    lum = a @ np.array([0.299, 0.587, 0.114])
    mask = np.clip((lum - 185) / 70.0, 0, 1)[..., None]
    glow = mask * np.array([255, 170, 90], dtype=float)
    glow_img = Image.fromarray(glow.astype("uint8")).filter(
        ImageFilter.GaussianBlur(radius=max(img.size) / 100.0)
    )
    g = np.asarray(glow_img, dtype=float)
    out = 255 - (255 - a) * (255 - g * strength) / 255.0  # screen blend
    return np.clip(out, 0, 255)


def add_grain(arr, sigma, rng):
    h, w = arr.shape[:2]
    noise = rng.normal(0, sigma, (h, w))
    return np.clip(arr + noise[..., None], 0, 255)


def vignette(arr, amount):
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    mask = 1 - (1 - amount) * np.clip(d / 1.42, 0, 1) ** 2
    return np.clip(arr * mask[..., None], 0, 255)


def process(path, p, outdir):
    img = Image.open(path).convert("RGB")
    rng = _rng(os.path.basename(path))
    arr = np.asarray(img, dtype=float)
    arr = grade(arr, p)
    arr = halation(Image.fromarray(arr.astype("uint8")), p["halation"])
    arr = add_grain(arr, p["grain_sigma"], rng)
    arr = vignette(arr, p["vignette"])
    out = Image.fromarray(arr.astype("uint8"))
    base, ext = os.path.splitext(os.path.basename(path))
    dest = os.path.join(outdir, f"{base}_graded{ext}")
    out.save(dest, quality=90)
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--preset", choices=list(PRESETS), default="ouatih")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    p = PRESETS[a.preset]
    os.makedirs(a.out, exist_ok=True)
    for path in a.inputs:
        if os.path.isfile(path):
            print(process(path, p, a.out))


if __name__ == "__main__":
    main()
