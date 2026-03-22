# ==========================================
# Stage 1: Build Frontend (Next.js Static Export)
# ==========================================
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend

# Copy frontend source
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ .
# Build the static export
RUN npm run build

# ==========================================
# Stage 2: Build Backend & Serve App
# ==========================================
FROM python:3.10-slim
WORKDIR /app

# Install system dependencies required for OpenCV/PyTorch if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces require user 1000
RUN useradd -m -u 1000 user
ENV HOME=/home/user
ENV PATH=/home/user/.local/bin:$PATH

WORKDIR /home/user/app

# Install Python requirements
COPY --chown=user backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code including trained checkpoints
COPY --chown=user backend/ backend/

# Copy the built static frontend from Stage 1 into the location expected by FastAPI
COPY --chown=user --from=frontend-builder /app/frontend/out frontend/out

# Switch to the non-root user
USER 1000

# Hugging Face Spaces expose port 7860
EXPOSE 7860

# Set environment variables
ENV HOST=0.0.0.0
ENV PORT=7860

WORKDIR /home/user/app/backend

# Start the unified FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
