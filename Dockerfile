FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    runpod==1.7.4 \
    diffusers>=0.38.0 \
    transformers>=4.48.0 \
    accelerate>=1.2.0 \
    safetensors>=0.4.5 \
    sentencepiece>=0.2.0 \
    protobuf>=5.28.3 \
    huggingface_hub>=0.27.0 \
    requests>=2.32.3 \
    Pillow>=11.0.0 \
    peft>=0.14.0 \
    bitsandbytes>=0.45.0 \
    ultralytics>=8.3.0 \
    opencv-python-headless>=4.10.0

# Model weights are loaded at runtime from either:
# 1. RunPod Network Volume (/runpod-volume/flux-dev/)
# 2. HuggingFace Hub (fallback, requires HF_TOKEN env var)

# Copy handler
COPY handler.py /app/handler.py

# RunPod serverless entry
CMD ["python3", "-u", "/app/handler.py"]
