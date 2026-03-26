FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    "fastapi>=0.110.3" \
    "uvicorn>=0.34.0" \
    "aiohttp~=3.13" \
    "pydantic>=2.9.2" \
    "python-dotenv>=1.0.0"

COPY . /app/gateway/

ENV PYTHONPATH=/app

EXPOSE 8090

CMD ["uvicorn", "gateway.app:app", "--host", "0.0.0.0", "--port", "8090", "--log-level", "info"]
