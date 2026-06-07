FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Upgrade torch (base has 2.4, need >=2.5 for diffusers FluxPipeline)
RUN pip install --no-cache-dir \
    torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Install Python dependencies
# diffusers 0.38.0+ has Flux2Pipeline (FLUX.2-dev)
# transformers 4.50+ has Mistral3, <4.52 avoids float8_e8m0fnu (torch 2.7 req)
RUN pip install --no-cache-dir \
    runpod \
    'diffusers>=0.38.0' \
    'transformers>=4.50.0,<4.52.0' \
    accelerate \
    safetensors \
    sentencepiece \
    protobuf \
    huggingface_hub \
    requests \
    Pillow \
    peft \
    'bitsandbytes>=0.43.0' \
    ultralytics \
    opencv-python-headless

# Verify critical imports work at build time
RUN python3 -c "from diffusers import Flux2Pipeline; from transformers import Mistral3ForConditionalGeneration; print('All Flux 2 imports OK')"

# Model weights are loaded at runtime from either:
# 1. RunPod Network Volume (/runpod-volume/flux-dev/)
# 2. HuggingFace Hub (fallback, requires HF_TOKEN env var)

# Copy handler
COPY handler.py /app/handler.py

# RunPod serverless entry
CMD ["python3", "-u", "/app/handler.py"]
