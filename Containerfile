# Base Image
FROM docker.io/pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# System Dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN echo 'APT::Sandbox::User "root";' > /etc/apt/apt.conf.d/sandbox-disable
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN conda install -y -c conda-forge git ninja wget

# Install Python Dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Specialized Libraries
RUN pip install flash-attn --no-build-isolation
RUN pip install torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.5.0+cu124.html

ENV CUDA_HOME=/usr/local/cuda

WORKDIR /workspace/libs
RUN git clone https://github.com/Pointcept/Pointcept.git && \
    cd Pointcept && \
    # Install dependencies first to be safe
    pip install --no-cache-dir -r requirements.txt && \
    # Install Pointcept (not editable, fully compiled)
    pip install -v .

WORKDIR /workspace/project