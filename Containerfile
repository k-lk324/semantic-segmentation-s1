# Base Image
FROM docker.io/pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

# System Dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN echo 'APT::Sandbox::User "root";' > /etc/apt/apt.conf.d/sandbox-disable

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsparsehash-dev \
    && rm -rf /var/lib/apt/lists/*

RUN conda install -y -c conda-forge git ninja wget

# Install Python Dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Specialized Libraries
RUN pip install flash-attn --no-build-isolation
RUN pip install torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.7.0+cu128.html

RUN pip install spconv-cu128
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0"

WORKDIR /workspace/project

