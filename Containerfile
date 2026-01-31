# Base Image
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# System Dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git \
    ninja-build \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Python Dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Specialized Libraries
RUN pip install flash-attn --no-build-isolation
RUN pip install torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.5.0+cu124.html

# Setup Workspace
WORKDIR /workspace/project