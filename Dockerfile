FROM python:3.11-slim

# Install system libraries required by OpenCV, Pillow, FAISS, and PyTorch CPU
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs as a non-root user with UID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR $HOME/app

# Install dependencies: first CPU PyTorch wheel, then remaining prod dependencies
COPY --chown=user requirements-prod.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements-prod.txt

# Copy application files
COPY --chown=user . .

# Ensure data directory exists with write permissions for SQLite checkpoints and case storage
RUN mkdir -p data/cases data/checkpoints

EXPOSE 7860

# Runs on PORT (defaults to 7860 for Hugging Face Spaces, or adapts to Cloud Run / Render)
CMD ["sh", "-c", "uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT:-7860}"]

