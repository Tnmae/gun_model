# Use NVIDIA's PyTorch image with CUDA 12.1
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# Install system deps (git is REQUIRED for torchreid source install)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy requirements first (for Docker cache)
COPY requirements.txt .

# Install Python dependencies (WITHOUT torchreid)
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---- install torchreid from source (Python 3.11 compatible) ----
RUN git clone https://github.com/KaiyangZhou/deep-person-reid.git /opt/torchreid \
    && cd /opt/torchreid \
    && pip install -r requirements.txt \
    && python setup.py install \
    && rm -rf /opt/torchreid

# Copy app source
COPY . .

# Expose port and run
EXPOSE 8004
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8004"]
