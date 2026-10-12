# All-in-one image: API + Streamlit UI. Works as a Hugging Face Docker Space as-is.
# For the API alone:  docker run -p 8000:8000 <image> uvicorn api.main:app --host 0.0.0.0 --port 8000
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
# Hugging Face Spaces run containers as uid 1000.
RUN useradd -m -u 1000 user
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HOME=/home/user HF_HOME=/home/user/.hf \
    QDRANT_PATH=/home/user/qdrant_data SEED_DIR=/app/data/corpus API_URL=http://127.0.0.1:8000 PORT=7860
WORKDIR /app

# CPU-only torch keeps the image ~1.5 GB smaller than the default CUDA build.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=user groundtruth ./groundtruth
COPY --chown=user api ./api
COPY --chown=user ui ./ui
COPY --chown=user data ./data
COPY --chown=user deploy ./deploy
USER user
# Bake the models into the image so cold starts don't download ~700 MB.
RUN python -c "from groundtruth.models import embedder, reranker, nli; embedder(); reranker(); nli()"

EXPOSE 7860
CMD ["sh", "deploy/start.sh"]
