FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential pkg-config \
    ffmpeg \
    libavdevice-dev libavfilter-dev libavformat-dev libavcodec-dev libswresample-dev libswscale-dev \
    libopus-dev libvpx-dev \
    libsrtp2-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch, installed before requirements.txt so Docker layer caching
# doesn't re-download it whenever an unrelated dependency changes. Without
# --index-url .../cpu, ultralytics pulls a multi-GB CUDA build for no benefit
# on Render's GPU-less instances.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server/ server/
COPY static/ static/
COPY models/ models/

EXPOSE 8000
CMD ["sh", "-c", "uvicorn server.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
