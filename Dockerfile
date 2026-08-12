FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.9.1+cpu && \
    pip install --no-cache-dir .

COPY configs ./configs
COPY examples ./examples

CMD ["vision-pipeline", "demo", "--config", "configs/demo.yaml"]
