FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# When building from repo root context, copy from server/
COPY server/requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install -r /app/requirements.txt

# Preserve the Python package structure under /app/server
COPY server/ /app/server

# Expose uvicorn port
EXPOSE 8000

# Render fournit $PORT. Utilisons-le (par défaut 8000 en local)
ENV PORT=8000
# Run uvicorn using the package path so relative imports (e.g., from .config) work
CMD ["sh", "-c", "python -m uvicorn server.main:app --host 0.0.0.0 --port ${PORT}"]


