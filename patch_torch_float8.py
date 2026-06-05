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

# Use globals() to avoid _C reference issues — float8_e4m3fn is available
# after the wildcard import, and globals() lets us add new names at module scope
patch = '''
# Patch: add missing float8 aliases for compat with latest transformers
for _f8_attr in ["float8_e8m0fnu", "float8_e4m3fnuz", "float8_e5m2fnuz"]:
    if _f8_attr not in dir():
        globals()[_f8_attr] = float8_e4m3fn
del _f8_attr
'''

content = content.replace(marker, marker + patch)
with open(init_file, 'w') as f:
    f.write(content)

print(f"Patched {init_file} with float8 aliases")
