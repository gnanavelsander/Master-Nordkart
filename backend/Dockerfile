FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U pip uv

COPY pyproject.toml uv.lock ./

# lager requirements inne i container (ikke i repo) og installerer i system python
RUN uv export --format requirements-txt --locked > /tmp/requirements.lock.txt \
 && uv pip install --system -r /tmp/requirements.lock.txt

COPY . .

EXPOSE 8000
CMD ["python", "-m", "server.app"]
