# --- Stage 1: Build Stage ---
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY main.cpp n_queen_gpu.cu n_queen_gpu.h ./
RUN nvcc -O3 -std=c++17 main.cpp n_queen_gpu.cu -o gpu_server

# --- Stage 2: Runtime Stage ---
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY --from=builder /src/gpu_server .
COPY handler.py requirements.txt ./

RUN pip3 install --no-cache-dir -r requirements.txt

# Unbuffered output forces Python to stream logs to stdout immediately
CMD ["python3", "-u", "handler.py"]