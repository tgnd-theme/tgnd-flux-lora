FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    runpod==1.7.4 \
    diffusers==0.31.0 \
    transformers==4.46.0 \
    accelerate==1.1.0 \
    safetensors==0.4.5 \
    sentencepiece==0.2.0 \
    protobuf==5.28.3 \
    huggingface_hub==0.26.2 \
    requests==2.32.3 \
    Pillow==11.0.0 \
    peft==0.13.2

# Model weights are loaded at runtime from either:
# 1. RunPod Network Volume (/runpod-volume/flux-dev/)
# 2. HuggingFace Hub (fallback, requires HF_TOKEN env var)

# Copy handler
COPY handler.py /app/handler.py

# RunPod serverless entry
CMD ["python3", "-u", "/app/handler.py"]
