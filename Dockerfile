FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.app.txt ./
RUN pip install --no-cache-dir -r requirements.app.txt

COPY . ./

RUN chmod +x /app/scripts/start.sh

EXPOSE 8001 8501

CMD ["/app/scripts/start.sh"]
