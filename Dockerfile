# syntax=docker/dockerfile:1

FROM python:3.12-slim-bookworm

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Dependency layer (cached when only app code changes)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Application (imports resolve from WORKDIR: models, env, server.app)
COPY models.py env.py agent.py ./
COPY server/ ./server/

EXPOSE 7860

# openenv.yaml: module server.app, attr app, port 7860
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
