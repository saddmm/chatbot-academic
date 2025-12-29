# Gunakan image Python 3.11 slim yang ringan
FROM python:3.11-slim

# Set working directory di dalam container
WORKDIR /app

# Set environment variables
# PYTHONDONTWRITEBYTECODE: Mencegah Python menulis file .pyc
# PYTHONUNBUFFERED: Memastikan output log Python langsung muncul (tidak di-buffer)
# PYTHONPATH: Menambahkan direktori kerja ke path agar module 'app' bisa diimport
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system dependencies yang mungkin dibutuhkan
# build-essential: untuk compile beberapa package python jika perlu
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt terlebih dahulu untuk memanfaatkan cache layer Docker
COPY requirements.txt .

# Install dependencies Python
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh source code project ke dalam container
COPY . .

# Expose port 5000 (port default Flask app Anda)
EXPOSE 5000

# Command untuk menjalankan aplikasi
CMD ["python", "api/app.py"]
