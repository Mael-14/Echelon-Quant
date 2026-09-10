FROM python:3.12-slim

WORKDIR /app
COPY backend /app/backend
WORKDIR /app/backend/apps/api-gateway

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH=/app/backend
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
