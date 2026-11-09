FROM python:3.11-slim

# Install system dependencies: ffmpeg (for audio/video), git, build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ ./backend/

# Expose the FastAPI port
EXPOSE 8000

# Set environment variables
ENV VIZ_TOKEN=stilleazy
ENV PYTHONUNBUFFERED=1

# Start the server
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
