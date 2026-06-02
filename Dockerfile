FROM python:3.12-slim

# Avoid writing .pyc files and buffering stdout/stderr.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first for better layer caching.
# We install the project (which pulls in its declared dependencies). The actual
# application source is bind-mounted in docker-compose for live reload, so the
# installed package is only used to resolve dependencies.
COPY pyproject.toml README.md ./
COPY app ./app
RUN pip install --no-cache-dir .

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
