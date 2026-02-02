FROM mcr.microsoft.com/devcontainers/python:3.12

RUN apt-get update && apt-get install -y --no-install-recommends \
    postgresql-client \
    docker.io \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U pip



RUN apt-get update && apt-get install -y --no-install-recommends \
    postgresql-client \
    docker-cli \
    && rm -rf /var/lib/apt/lists/*




COPY requirements.txt .
RUN pip install -r requirements.txt



