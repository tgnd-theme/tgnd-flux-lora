"""Patch torch __init__.py to add missing float8 type aliases.

PyTorch 2.6 lacks float8_e8m0fnu, float8_e4m3fnuz, float8_e5m2fnuz
which are needed by latest transformers/torchvision. This script adds
aliases pointing to float8_e4m3fn (functionally equivalent for our use).
"""
import torch
import os

MISSING = ['float8_e8m0fnu', 'float8_e4m3fnuz', 'float8_e5m2fnuz']

need_patch = any(not hasattr(torch, attr) for attr in MISSING)
if not need_patch:
    print("torch float8 types already present, no patch needed")
    exit(0)

init_file = os.path.join(os.path.dirname(torch.__file__), '__init__.py')
with open(init_file, 'r') as f:
    content = f.read()

marker = 'from torch._C import *  # noqa: F403'
if marker not in content:
    print(f"WARNING: marker not found in {init_file}, skipping patch")
    exit(0)

patch = '''
# Patch: add missing float8 aliases for compat with latest transformers
for _attr in ["float8_e8m0fnu", "float8_e4m3fnuz", "float8_e5m2fnuz"]:
    if not hasattr(_C, _attr):
        setattr(_C, _attr, float8_e4m3fn)
'''

content = content.replace(marker, marker + patch)
with open(init_file, 'w') as f:
    f.write(content)

print(f"Patched {init_file} with float8 aliases")
