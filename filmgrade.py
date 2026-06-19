#!/usr/bin/env python3
"""
filmgrade.py - Bakt de Once Upon a Time-look in foto's + anti-AI processing.

Warme, gouden, vivid en SCHONE 35mm-grade (Once Upon a Time in Hollywood).
Deterministisch per bestand (seed = hash van bestandsnaam): dezelfde foto geeft
altijd hetzelfde resultaat, verschillende foto's krijgen net andere fijne korrel.

Anti-AI laag: breekt de typische AI-glans door micro-texture, chromatische
aberratie, lokale scherpte variatie, sensor noise patroon, en JPEG compressie.
Maakt AI-gegenereerde foto's ononderscheidbaar van echte foto's.

Gebruik:
    python3 filmgrade.py foto.jpg                  # -> foto_graded.jpg
    python3 filmgrade.py *.jpg --out verwerkt/      # batch
    python3 filmgrade.py foto.jpg --preset zishy    # nog lichtere touch
    python3 filmgrade.py foto.jpg --no-deai         # alleen grade, geen anti-AI

Presets:
    ouatih (default)  Once Upon a Time in Hollywood: warm, gouden, vivid, SCHOON.
                      De default voor dame-foto's. Matcht Zishy, poetst de
                      echtheid (huidtextuur, tan lines) niet weg.
    zishy             nog lichtere touch, voor een al sterk gouden harde-zon shot
                      waar ouatih te veel wordt.
"""

import io, os, argparse, hashlib
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

# Anti-AI parameters — breekt de perfecte AI-glans
DEAI = dict(
    # Micro-texture: fijne huidachtige textuur over het beeld
    micro_texture_sigma=4.0,     # sterkte van micro-texture noise (was 2.5)
    micro_texture_scale=3,       # schaal van textuurpatroon (pixels)

    # Chromatische aberratie: lichte kleurverschuiving aan beeldranden
    chroma_aberration=1.5,       # pixels verschuiving aan uiterste rand (was 1.2)

    # Lokale scherpte variatie: niet alles pin-sharp
    local_blur_strength=0.35,    # blend met licht geblurde versie (was 0.3)
    local_blur_radius=1.2,       # radius van de zachte blur (was 1.0)

    # Sensor noise: per-kanaal noise zoals echte camera sensor
    sensor_noise_r=4.5,          # rood kanaal noise sigma (was 3.0)
    sensor_noise_g=3.0,          # groen kanaal noise (was 2.0)
    sensor_noise_b=5.0,          # blauw kanaal noise (was 3.5)

    # Hot pixels: zeldzame heldere puntjes zoals echte sensoren
    hot_pixel_chance=0.00005,    # kans per pixel (was 0.00003)

    # Luminance-dependent noise: meer noise in schaduwen (zoals echte camera)
    shadow_noise_boost=2.5,      # extra noise in donkere gebieden (was 2.0)

    # JPEG compressie: echte foto's hebben altijd JPEG artifacts
    jpeg_quality=75,             # opslaan op 75 ipv 82 — meer compressie = minder AI-smooth
)


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


# ============== WATERMARK REMOVAL ==============

def remove_watermark(img, margin_bottom=50, margin_right=180):
    """Verwijdert het Zishy watermark uit de rechteronderhoek.

    Kopieert het blok pixels van net boven het watermark naar de
    watermark-positie, met een zachte verticale blend voor naadloze
    overgang. Werkt op elke foto met het standaard Zishy watermark.
    """
    arr = np.asarray(img, dtype=float).copy()
    h, w = arr.shape[:2]
    y_start = max(0, h - margin_bottom)
    x_start = max(0, w - margin_right)
    patch_h = h - y_start
    # Kopieer het blok van net boven de watermark regio
    src_y = max(0, y_start - patch_h)
    patch = arr[src_y:y_start, x_start:w, :].copy()
    # Flip verticaal zodat de overgang bij y_start naadloos is
    patch = patch[::-1]
    # Zachte blend: bovenaan (bij y_start) 100% patch, onderaan 90% patch
    blend = np.linspace(1.0, 0.9, patch_h)[:, None, None]
    arr[y_start:h, x_start:w, :] = patch * blend + arr[y_start:h, x_start:w, :] * (1 - blend)
    return Image.fromarray(np.clip(arr, 0, 255).astype("uint8"))


# ============== ANTI-AI FUNCTIES ==============

def micro_texture(arr, rng, d):
    """Voegt fijne huidachtige micro-textuur toe.

    Verschilt van gewone grain: dit is een gestructureerd patroon op kleine
    schaal dat lijkt op huidporiën en fijne textuurdetails die AI mist.
    """
    h, w = arr.shape[:2]
    scale = d["micro_texture_scale"]
    # Maak textuur op lagere resolutie en schaal op — geeft structuur ipv ruis
    th, tw = h // scale, w // scale
    texture = rng.normal(0, d["micro_texture_sigma"], (th, tw))
    # Opschalen met nearest neighbor behoudt blokachtige textuurstructuur
    texture_full = np.repeat(np.repeat(texture, scale, axis=0), scale, axis=1)
    texture = np.zeros((h, w), dtype=texture_full.dtype)
    th2, tw2 = min(h, texture_full.shape[0]), min(w, texture_full.shape[1])
    texture[:th2, :tw2] = texture_full[:th2, :tw2]
    # Sterker in middentonen (huidgebieden), zwakker in highlights/schaduwen
    lum = arr @ np.array([0.299, 0.587, 0.114])
    midtone_mask = np.clip(1.0 - np.abs(lum - 128) / 100.0, 0.2, 1.0)
    return np.clip(arr + texture[..., None] * midtone_mask[..., None], 0, 255)


def chromatic_aberration(arr, d):
    """Verschuift R en B kanalen licht naar buiten — zoals een echt objectief.

    Elk objectief heeft chromatische aberratie. AI-beelden hebben het niet.
    Dit is een van de sterkste signalen voor "echt vs AI".
    """
    h, w = arr.shape[:2]
    shift = d["chroma_aberration"]
    if shift < 0.1:
        return arr

    # Afstand van centrum (0 in midden, 1 aan rand)
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2, w / 2
    dist = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)

    # Per-pixel verschuiving proportioneel aan afstand van centrum
    shift_map = dist * shift

    # Verschuif rood kanaal naar buiten (weg van centrum)
    dx_r = (xx - cx) / np.maximum(np.sqrt((xx - cx)**2 + (yy - cy)**2), 1) * shift_map
    dy_r = (yy - cy) / np.maximum(np.sqrt((xx - cx)**2 + (yy - cy)**2), 1) * shift_map

    # Gebruik simpele nearest-neighbor lookup voor snelheid
    src_x_r = np.clip(np.round(xx - dx_r).astype(int), 0, w - 1)
    src_y_r = np.clip(np.round(yy - dy_r).astype(int), 0, h - 1)
    src_x_b = np.clip(np.round(xx + dx_r).astype(int), 0, w - 1)
    src_y_b = np.clip(np.round(yy + dy_r).astype(int), 0, h - 1)

    out = arr.copy()
    out[..., 0] = arr[src_y_r, src_x_r, 0]  # rood naar buiten
    out[..., 2] = arr[src_y_b, src_x_b, 2]  # blauw naar binnen
    return out


def local_sharpness_variation(arr, d):
    """Blend met licht geblurde versie — niet alles pin-sharp.

    Echte foto's hebben depth of field: achtergrond en randen zijn zachter.
    AI maakt alles even scherp. Deze functie simuleert een subtiel DOF-effect.
    """
    h, w = arr.shape[:2]
    strength = d["local_blur_strength"]
    radius = d["local_blur_radius"]
    if strength < 0.01:
        return arr

    img = Image.fromarray(arr.astype("uint8"))
    blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))
    blurred_arr = np.asarray(blurred, dtype=float)

    # Meer blur aan randen, minder in centrum (simuleert DOF)
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    blur_mask = np.clip(dist * 1.5 - 0.3, 0, 1) * strength
    blur_mask = blur_mask[..., None]

    return np.clip(arr * (1 - blur_mask) + blurred_arr * blur_mask, 0, 255)


def sensor_noise(arr, rng, d):
    """Per-kanaal sensor noise zoals een echte camera.

    Echte sensors hebben meer noise in blauw (minder photosites), minder in
    groen (meeste photosites). AI-beelden hebben uniforme, schone kanalen.
    Noise is ook sterker in schaduwen (shot noise / read noise).
    """
    h, w = arr.shape[:2]
    lum = arr @ np.array([0.299, 0.587, 0.114])

    # Luminance-dependent noise: meer in schaduwen
    shadow_factor = 1.0 + (d["shadow_noise_boost"] - 1.0) * np.clip(1.0 - lum / 100.0, 0, 1)

    out = arr.copy()
    for ch, sigma in enumerate([d["sensor_noise_r"], d["sensor_noise_g"], d["sensor_noise_b"]]):
        noise = rng.normal(0, sigma, (h, w)) * shadow_factor
        out[..., ch] = np.clip(out[..., ch] + noise, 0, 255)

    return out


def hot_pixels(arr, rng, d):
    """Zeldzame heldere puntjes — elke echte sensor heeft ze.

    Dit zijn stuck pixels die altijd helder zijn. Heel subtiel maar het
    breekt de perfectie van AI-beelden.
    """
    h, w = arr.shape[:2]
    chance = d["hot_pixel_chance"]
    mask = rng.random((h, w)) < chance
    if not mask.any():
        return arr
    out = arr.copy()
    # Hot pixels zijn helder in een willekeurig kanaal
    channels = rng.integers(0, 3, size=mask.sum())
    ys, xs = np.where(mask)
    for i, (y, x) in enumerate(zip(ys, xs)):
        out[y, x, channels[i]] = min(out[y, x, channels[i]] + 80, 255)
    return out


def jpeg_compress(img, quality):
    """Sla op als JPEG en lees terug — voegt echte JPEG artifacts toe.

    Elke foto op internet is JPEG-gecomprimeerd. AI-output is dat niet.
    JPEG artifacts (blokranden, kleurverlies) zijn een sterk "echt" signaal.
    """
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()


def deai(arr, rng, d):
    """Volledige anti-AI processing pipeline."""
    arr = micro_texture(arr, rng, d)
    arr = chromatic_aberration(arr, d)
    arr = local_sharpness_variation(arr, d)
    arr = sensor_noise(arr, rng, d)
    arr = hot_pixels(arr, rng, d)
    return arr


# ============== PROCESS ==============

def process(path, p, outdir, apply_deai=True, strip_watermark=True):
    img = Image.open(path).convert("RGB")
    if strip_watermark:
        img = remove_watermark(img)
    rng = _rng(os.path.basename(path))
    arr = np.asarray(img, dtype=float)

    # Stap 1: Filmgrade (kleur, warmte, korrel)
    arr = grade(arr, p)
    arr = halation(Image.fromarray(arr.astype("uint8")), p["halation"])
    arr = add_grain(arr, p["grain_sigma"], rng)
    arr = vignette(arr, p["vignette"])

    # Stap 2: Anti-AI processing
    if apply_deai:
        arr = deai(arr, rng, DEAI)

    out = Image.fromarray(arr.astype("uint8"))

    # Stap 3: JPEG compressie (alleen met anti-AI)
    if apply_deai:
        out = jpeg_compress(out, DEAI["jpeg_quality"])

    base, ext = os.path.splitext(os.path.basename(path))
    dest = os.path.join(outdir, f"{base}_graded{ext}")
    out.save(dest, quality=DEAI["jpeg_quality"] if apply_deai else 90)
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--preset", choices=list(PRESETS), default="ouatih")
    ap.add_argument("--out", default=".")
    ap.add_argument("--no-deai", action="store_true",
                    help="Alleen filmgrade, geen anti-AI processing")
    ap.add_argument("--keep-watermark", action="store_true",
                    help="Zishy watermark NIET verwijderen")
    a = ap.parse_args()
    p = PRESETS[a.preset]
    os.makedirs(a.out, exist_ok=True)
    for path in a.inputs:
        if os.path.isfile(path):
            print(process(path, p, a.out,
                          apply_deai=not a.no_deai,
                          strip_watermark=not a.keep_watermark))


if __name__ == "__main__":
    main()
