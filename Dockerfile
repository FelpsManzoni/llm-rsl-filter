FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV APP_DIR=/app
ENV WORK_DIR=/work

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY src /app/src
COPY README.md /app/README.md
COPY .env.example /app/.env.example

RUN mkdir -p /work /work/data /work/output
VOLUME ["/work", "/work/data", "/work/output"]

WORKDIR /work
ENV PYTHONPATH=/app

CMD ["python", "-m", "src.main"]
