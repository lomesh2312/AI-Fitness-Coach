# Use a lightweight base image to keep the container small and fast
FROM python:3.10-slim

# Prevent Python from writing .pyc files & buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Copy only the requirements first, to leverage Docker cache
COPY requirements.txt .

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies (no-cache-dir reduces image size)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Change working directory so all relative imports work seamlessly
WORKDIR /app/backend

# Start uvicorn. Note it uses $PORT to dynamically grab Render's assigned port
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
