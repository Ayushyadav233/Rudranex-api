# Base Image
FROM python:3.9-slim

# System Dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglx0 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Sabse pehle pip aur essential tools update karo
RUN pip install --upgrade pip setuptools wheel

COPY requirements.txt .

# Protobuf aur NumPy ko requirements se pehle force install karo
RUN pip install "numpy<2.0.0" "protobuf==3.20.*"

# Baaki requirements install karo
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]