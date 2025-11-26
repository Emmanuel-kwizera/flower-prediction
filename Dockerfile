FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Fix for macOS/Linux TensorFlow/OpenMP runtime conflict (safe for container)
ENV KMP_DUPLICATE_LIB_OK=True
ENV OMP_NUM_THREADS=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY api/ ./api/
COPY scripts/ ./scripts/
COPY web/ ./web/
# Copy models if they exist (will fail if directory doesn't exist, so we use a wildcard hack or ensure it exists)
# Using COPY . . is easier but copies venv/ and data/ which we might not want.
# Let's copy specific directories.
COPY models/ ./models/

# Create directories that might be missing but needed
RUN mkdir -p data visualizations

# Expose port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
