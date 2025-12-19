FROM python:3.10-slim

WORKDIR /app

# System dependencies (needed for Pillow & TensorFlow)
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Hugging Face Spaces port
EXPOSE 7860

# Run Flask app
CMD ["python", "app.py"]
