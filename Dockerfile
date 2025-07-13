# Use a base Python image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies (including Rust and build tools)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        gcc \
        pkg-config \
        libssl-dev \
        libffi-dev \
        python3-dev \
        rustc \
        cargo \
        wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY ./py/requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy FastAPI app source code
COPY ./py /app

# Expose the port FastAPI will run on
EXPOSE 10000

# Start FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
