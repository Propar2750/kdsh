FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

WORKDIR /app

# Install Python and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install pathway and dependencies with CUDA 12.8 PyTorch (nightly for Blackwell support)
RUN python -m pip install --upgrade pip && \
    python -m pip install pathway pandas pytest numpy && \
    python -m pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128 && \
    python -m pip install sentence-transformers

# Copy project files
COPY . .

# Default command
CMD ["python", "-m", "pytest", "tests/", "-v"]
