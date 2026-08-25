FROM python:3.11-slim AS builder
WORKDIR /install
COPY app/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install/deps -r requirements.txt

FROM python:3.11-slim AS runtime
WORKDIR /app
COPY --from=builder /install/deps /usr/local
COPY app/main.py app/registry.json ./
ENV PYTHONDONTWRITEBYTECODE=1
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]