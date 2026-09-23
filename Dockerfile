# Сервис агента прогноза ВЭС (API + Swagger на /docs).
# Сборка и запуск:  docker build -t wind-agent . && docker run -p 8000:8000 wind-agent
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=8000 AGENT_MODEL=lightgbm AGENT_SCHEDULE_HOURS=6
EXPOSE 8000
CMD ["sh", "-c", "python -m uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
