# syntax=docker/dockerfile:1
# docker build -t novapro/gym-fitsight . && docker tag novapro/gym-fitsight registry.xnet.com:5000/novapro/gym-fitsight:latest && docker push registry.xnet.com:5000/novapro/gym-fitsight

ARG BASE_IMAGE=pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics

WORKDIR /app

# System deps for OpenCV + moviepy/ffmpeg + downloads (+file for debugging)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    ca-certificates \
    curl \
    file \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy project
COPY . /app

RUN mkdir -p /app/cache /app/cache/analysis /app/models /app/static/output /tmp/Ultralytics

# Upgrade packaging tooling
RUN python -m pip install --upgrade pip setuptools wheel

# Install deps without downgrading torch/torchvision from the base image
RUN awk '!/^torch(vision)?(>=|==|!=|<|>|~=)/{print}' requirements.txt > /tmp/requirements.no-torch.txt && \
    python -m pip install --prefer-binary -r /tmp/requirements.no-torch.txt

ENV YOLO_WEIGHTS=/app/yolov7-w6-pose.pt

EXPOSE 3000

CMD ["python", "app.py"]
