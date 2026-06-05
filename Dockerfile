FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Upgrade torch first (base has 2.4, need 2.6 for diffusers compat)
RUN pip install --no-cache-dir \
    torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Install Python dependencies
RUN pip install --no-cache-dir \
    runpod \
    diffusers \
    transformers \
    accelerate \
    safetensors \
    sentencepiece \
    protobuf \
    huggingface_hub \
    requests \
    Pillow \
    peft \
    bitsandbytes \
    ultralytics \
    opencv-python-headless

# Patch missing float8 types (added in PyTorch 2.7, not in 2.6)
RUN python3 -c "\
import torch; \
patch = False; \
for attr in ['float8_e8m0fnu', 'float8_e4m3fnuz', 'float8_e5m2fnuz']: \
    if not hasattr(torch, attr): \
        patch = True; break; \
if patch: \
    import torch as _t; \
    init = _t.__file__.replace('__init__.py', '') + '__init__.py'; \
    with open(init, 'r') as f: lines = f.readlines(); \
    marker = 'from torch._C import *  # noqa: F403'; \
    new_lines = []; \
    for line in lines: \
        new_lines.append(line); \
        if marker in line: \
            new_lines.append('# Patch: add missing float8 aliases for compat with latest transformers\n'); \
            new_lines.append('for _attr in [\"float8_e8m0fnu\", \"float8_e4m3fnuz\", \"float8_e5m2fnuz\"]:\n'); \
            new_lines.append('    if not hasattr(_C, _attr): setattr(_C, _attr, float8_e4m3fn)\n'); \
    with open(init, 'w') as f: f.writelines(new_lines); \
    print('Patched torch __init__.py with float8 aliases'); \
"

# Verify critical imports work at build time
RUN python3 -c "\
from transformers import CLIPImageProcessor; \
from diffusers import FluxPipeline, FluxTransformer2DModel, BitsAndBytesConfig; \
print('All imports OK'); \
import transformers, diffusers, torch; \
print(f'torch={torch.__version__} transformers={transformers.__version__} diffusers={diffusers.__version__}')"

# Model weights are loaded at runtime from either:
# 1. RunPod Network Volume (/runpod-volume/flux-dev/)
# 2. HuggingFace Hub (fallback, requires HF_TOKEN env var)

# Copy handler
COPY handler.py /app/handler.py

# RunPod serverless entry
CMD ["python3", "-u", "/app/handler.py"]
