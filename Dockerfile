# Magnet Eye backend. Target: a 1-core, 1 GB VPS (PR-13).
#   docker build -t magnet-eye-backend .
#   docker run --cpus=1 --memory=1g -p 8001:8001 magnet-eye-backend
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # One BLAS thread: the server has one core and runs one fit at a time.
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /srv
COPY pyproject.toml README.md LICENSE ./
COPY app ./app
RUN pip install --no-cache-dir . && useradd --system --no-create-home magnet
USER magnet

EXPOSE 8001
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/v1/health', timeout=4)"
# One process: sessions live in this process's memory, and fits run on its single worker.
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1", "--no-access-log"]
