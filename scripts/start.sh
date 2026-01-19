#!/usr/bin/env bash
set -euo pipefail

API_PORT=${API_PORT:-8001}
UI_PORT=${UI_PORT:-8501}

export API_URL="http://localhost:${API_PORT}"

uvicorn app.main:app --host 0.0.0.0 --port "${API_PORT}" &

streamlit run app/streamlit_app.py \
  --server.address 0.0.0.0 \
  --server.port "${UI_PORT}" \
  --server.headless true
