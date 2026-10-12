#!/bin/sh
# All-in-one container: FastAPI on 8000 (internal) + Streamlit on $PORT (public).
set -e
uvicorn api.main:app --host 127.0.0.1 --port 8000 &
API_PID=$!

echo "Waiting for the API (loading models, indexing sample docs)..."
until curl -sf http://127.0.0.1:8000/health > /dev/null; do
  kill -0 $API_PID 2>/dev/null || { echo "API failed to start"; exit 1; }
  sleep 2
done

exec streamlit run ui/app.py \
  --server.port "${PORT:-7860}" --server.address 0.0.0.0 --server.headless true \
  --server.enableCORS false --server.enableXsrfProtection false \
  --browser.gatherUsageStats false
