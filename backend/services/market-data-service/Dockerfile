FROM python:3.12-slim

WORKDIR /app
COPY . /app
WORKDIR /app/backend/services/market-data-service

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH=/app
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
