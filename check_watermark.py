#!/usr/bin/env python3
"""
Standalone tool to check whether a photo carries a Cabine42 invisible watermark, and — if you know
which session/attachment id you're checking against — whether it matches.

This does NOT require a GPU; it's plain CPU image processing (same watermark_utils.py used by the
generation handler). Run it locally or on any machine with the dependencies installed:

    pip install --no-deps invisible-watermark
    pip install PyWavelets opencv-python-headless pillow numpy torch

Usage:
    python3 check_watermark.py suspect_photo.jpg
    python3 check_watermark.py suspect_photo.jpg --expect "session-12345-attachment-9"

See watermark_utils.py for the honesty notes on what this can and cannot detect (survives direct
re-shares and moderate recompression; will usually NOT survive a screenshot or heavy recompression —
in that case this script will correctly report "no verifiable watermark found", not a false match).
"""

import argparse
import sys

from PIL import Image

from watermark_utils import decode_watermark, make_id_from_string, id_to_hex


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image_path", help="Path to the suspect photo (jpg/png)")
    parser.add_argument("--expect", help="Session/attachment id string to check against (optional)")
    args = parser.parse_args()

    try:
        img = Image.open(args.image_path)
    except Exception as e:
        print(f"Could not open image: {e}", file=sys.stderr)
        sys.exit(1)

    found = decode_watermark(img)

    if found is None:
        print("No verifiable watermark found.")
        print("This means either: the photo never had one, or it's been resized/recompressed enough")
        print("to destroy it (common after a screenshot or re-upload). This is NOT proof the photo")
        print("didn't originate here — only that the mark can't be verified in this copy.")
        sys.exit(2)

    found_hex = id_to_hex(found)
    print(f"Watermark found: {found_hex}")

    if args.expect:
        expected = make_id_from_string(args.expect)
        expected_hex = id_to_hex(expected)
        if found == expected:
            print(f"MATCH — this photo was generated for: {args.expect}")
        else:
            print(f"NO MATCH — found {found_hex}, expected {expected_hex} (for '{args.expect}')")
            print("This photo carries a Cabine42 watermark, but not from the session/id you checked.")


if __name__ == "__main__":
    main()
