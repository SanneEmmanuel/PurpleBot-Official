# Use slim Python base image
FROM python:3.10-slim-bullseye

# Set working directory inside container
WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libffi-dev \
        libssl-dev \
        curl \
        wget \
        git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY ./py/requirements.txt ./requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI app
COPY ./py /app

# Expose FastAPI port (HTTP + WebSocket on same endpoint)
EXPOSE 10000

# Run FastAPI app with Uvicorn (optimized)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000", "--workers", "1"]
