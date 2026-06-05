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
COPY patch_torch_float8.py /tmp/patch_torch_float8.py
RUN python3 /tmp/patch_torch_float8.py

# Verify critical imports work at build time
RUN python3 -c "from transformers import CLIPImageProcessor; from diffusers import FluxPipeline; print('All imports OK')"

# Model weights are loaded at runtime from either:
# 1. RunPod Network Volume (/runpod-volume/flux-dev/)
# 2. HuggingFace Hub (fallback, requires HF_TOKEN env var)

# Copy handler
COPY handler.py /app/handler.py

# RunPod serverless entry
CMD ["python3", "-u", "/app/handler.py"]
