FROM python:3.11-slim AS builder

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml setup.py MANIFEST.in ./
COPY __init__.py cli.py config.py exceptions.py launch.py deploy_to_colab.py epsionic_agent_complete.py ./
COPY core/ core/
COPY memory/ memory/
COPY tools/ tools/
COPY utils/ utils/
COPY tests/ tests/
COPY epsionic/ epsionic/
RUN pip install --no-cache-dir build && python -m build

FROM python:3.11-slim

LABEL org.opencontainers.image.source="https://github.com/epsionic/epsionic"
LABEL org.opencontainers.image.description="Epslionic-Colab: Production LLM Training Agent"
LABEL org.opencontainers.image.licenses="MIT"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl && pip install --no-cache-dir fastapi uvicorn

RUN addgroup --system epsionic && adduser --system --ingroup epsionic epsionic
USER epsionic
WORKDIR /workspace

VOLUME ["/workspace/data", "/workspace/config", "/workspace/logs"]

ENTRYPOINT ["epsionic"]
CMD ["--help"]
