FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src:/app/scripts \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

WORKDIR /app

RUN addgroup --system app && adduser --system --ingroup app app

# onnxruntime's StringNormalizer op (used by the converted TfidfVectorizer)
# needs a real system locale at session-init time -- python:3.12-slim ships
# none, which fails ONNX inference with a locale::facet error otherwise.
RUN apt-get update \
    && apt-get install -y --no-install-recommends locales \
    && sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen \
    && locale-gen \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --requirement requirements.txt

# One image serves the API (src/) and the whole training/optimization
# pipeline (scripts/): fastapi, scikit-learn, skl2onnx and onnxruntime all
# coexist in requirements.txt, so docker-compose's `trainer` and `api`
# services both run this image with a different command instead of
# maintaining separate images.
COPY --chown=app:app src ./src
COPY --chown=app:app scripts ./scripts

# Pre-create the mount points owned by `app`: named volumes (data/, models/,
# shared between the trainer and api services) are created root-owned on
# first use, and Docker seeds a fresh volume from whatever already exists at
# the mount point in the image — so this is what gives the non-root `app`
# user write access to them.
RUN mkdir -p /app/data /app/models && chown -R app:app /app/data /app/models

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"]

CMD ["uvicorn", "triage_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
