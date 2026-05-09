# Base Image
FROM python:3.9-slim

# System Dependencies (Build Error Fix)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglx0 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Requirements install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Project files copy
COPY . .

# Port expose (HF Standard)
EXPOSE 7860

# CMD with host and port binding
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]