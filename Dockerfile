FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update -o Acquire::Retries=5 \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 \
        libxcb1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.app.txt ./
RUN pip install --no-cache-dir -r requirements.app.txt

COPY . ./

RUN chmod +x /app/scripts/start.sh

EXPOSE 8001 8501

CMD ["/app/scripts/start.sh"]
