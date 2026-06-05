"""Patch torch to add missing float8 type aliases at runtime.

PyTorch 2.6 lacks float8_e8m0fnu, float8_e4m3fnuz, float8_e5m2fnuz
which are needed by latest transformers/torchvision. Instead of patching
the source file (fragile), we create a sitecustomize.py that patches
torch on import.
"""
import torch
import os
import sys

MISSING = ['float8_e8m0fnu', 'float8_e4m3fnuz', 'float8_e5m2fnuz']

need_patch = any(not hasattr(torch, attr) for attr in MISSING)
if not need_patch:
    print("torch float8 types already present, no patch needed")
    exit(0)

# Find site-packages dir
site_packages = None
for p in sys.path:
    if 'site-packages' in p or 'dist-packages' in p:
        if os.path.isdir(p):
            site_packages = p
            break

if not site_packages:
    print("WARNING: could not find site-packages dir")
    exit(1)

# Write a .pth file that executes on Python startup (before any imports)
pth_file = os.path.join(site_packages, 'torch_float8_patch.pth')
with open(pth_file, 'w') as f:
    f.write('import torch; [setattr(torch, a, torch.float8_e4m3fn) for a in ["float8_e8m0fnu","float8_e4m3fnuz","float8_e5m2fnuz"] if not hasattr(torch, a)]\n')

print(f"Created {pth_file}")

# Also patch the current process for the verification step
for attr in MISSING:
    if not hasattr(torch, attr):
        setattr(torch, attr, torch.float8_e4m3fn)

print("float8 patch applied")
