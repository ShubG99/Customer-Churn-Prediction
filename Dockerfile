# Multi-stage production Dockerfile for Telecom Customer Churn Prediction API
FROM python:3.11-slim as base

# Prevent Python from writing .pyc and buffer stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app"

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and configs
COPY configs/ ./configs/
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY models/ ./models/
COPY README.md MODEL_CARD.md ./

# Expose FastAPI default port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default execution
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
