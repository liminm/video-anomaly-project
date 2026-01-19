#!/usr/bin/env bash
set -euo pipefail

API_PORT=${API_PORT:-8001}
UI_PORT=${UI_PORT:-8501}

export API_URL="http://127.0.0.1:${API_PORT}"

uvicorn app.main:app --host 0.0.0.0 --port "${API_PORT}" &

python - <<'PYWAIT'
import os
import socket
import time

host = "127.0.0.1"
port = int(os.getenv("API_PORT", "8001"))

for _ in range(30):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        try:
            sock.connect((host, port))
            break
        except OSError:
            time.sleep(1)
PYWAIT

streamlit run app/streamlit_app.py \
  --server.address 0.0.0.0 \
  --server.port "${UI_PORT}" \
  --server.headless true
