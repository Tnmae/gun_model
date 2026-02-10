# ===============================
# Base image: PyTorch + CUDA 12.1
# ===============================
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# -------------------------------
# Environment settings
# -------------------------------
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# -------------------------------
# System dependencies
# -------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------
# App directory
# -------------------------------
WORKDIR /app

# -------------------------------
# Install Python dependencies
# (cached unless requirements.txt changes)
# -------------------------------
COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# -------------------------------
# Copy application source
# -------------------------------
COPY . /app

# -------------------------------
# Runtime
# -------------------------------
EXPOSE 8004

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8004"]
