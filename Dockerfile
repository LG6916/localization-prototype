FROM python:3.11-slim

# Open3D and trimesh need a few system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 libegl1 libx11-6 libxrender1 \
    libusb-1.0-0 curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

# Non-root user for on-prem safety
RUN useradd -ms /bin/bash appuser \
 && chown -R appuser:appuser /app \
 && mkdir -p /app/cache /app/uploads \
 && chown -R appuser:appuser /app/cache /app/uploads
USER appuser

EXPOSE 8050
ENV PYTHONUNBUFFERED=1 \
    DASH_HOST=0.0.0.0 \
    DASH_PORT=8050

CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "8050"]
