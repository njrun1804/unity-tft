# Dockerfile for LSTM training (CPU & GPU compatible)
FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy requirements
COPY requirements.txt ./

# Install Python dependencies (CPU by default)
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Optional: For GPU, user can override base image or install torch with CUDA
# Example: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Copy code
COPY . .

# Entrypoint (can override in CI)
ENTRYPOINT ["python", "scripts/train_lstm.py"]

# Default command (show help)
CMD ["--help"]
