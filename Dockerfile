# syntax=docker/dockerfile:1

FROM python:3.12-slim-bookworm

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Dependency layer (cached when only app code changes)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY models.py env.py agent.py grader.py inference.py ./
COPY openenv.yaml pyproject.toml ./
COPY tasks/ ./tasks/
COPY server/ ./server/
COPY tests/ ./tests/

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
