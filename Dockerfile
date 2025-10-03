# ==============================
# 1. Base image
# ==============================
FROM python:3.13-slim

# ==============================
# 2. Environment variables
# ==============================
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# ==============================
# 3. Work directory
# ==============================
WORKDIR /app

# ==============================
# 4. System dependencies
# ==============================
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# ==============================
# 5. Copy requirements & install
# ==============================
COPY requirements.txt /app/
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# ==============================
# 6. Copy project files and data
# ==============================
COPY Factor_Lens /app/Factor_Lens
COPY data /app/data

# ==============================
# 7. Run FastAPI using dynamic PORT
# ==============================
CMD ["sh", "-c", "uvicorn Factor_Lens.api:app --host 0.0.0.0 --port $PORT"]