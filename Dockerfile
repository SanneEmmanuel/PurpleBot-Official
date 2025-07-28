# Use slim base image
FROM python:3.10-slim-bullseye

# Set working directory
WORKDIR /app

# Install system dependencies + Node.js 20.x
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libffi-dev \
        libssl-dev \
        curl \
        wget \
        git \
        ca-certificates && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY ./py/requirements.txt ./requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY ./py /app
COPY ./js/main.js /app/main.js

# Expose ports for both FastAPI and Node.js socket server
EXPOSE 10000 3000

# Start both servers in a single CMD using bash
CMD bash -c "uvicorn main:app --host 0.0.0.0 --port 10000 --workers 1 & node /app/main.js"
