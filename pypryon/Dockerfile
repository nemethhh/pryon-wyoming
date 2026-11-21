# Build context: the repository root.
ARG BUILD_FROM=ghcr.io/home-assistant/amd64-base-debian:trixie-2026.11.0

FROM ghcr.io/astral-sh/uv:0.13.5 AS uv

FROM $BUILD_FROM AS build
RUN apt-get update && apt-get install -y --no-install-recommends python3 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /src
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_PYTHON=/usr/bin/python3 UV_PYTHON_DOWNLOADS=never UV_COMPILE_BYTECODE=1
RUN uv sync --frozen --no-dev --no-editable

FROM $BUILD_FROM
RUN apt-get update && apt-get install -y --no-install-recommends python3 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/venv /opt/venv
COPY pypryon/run.sh /run.sh
RUN chmod a+x /run.sh
LABEL org.opencontainers.image.source="https://github.com/nemethhh/pryon-wyoming" \
      org.opencontainers.image.description="Amazon Pryon wake words for Home Assistant (Wyoming); models not included" \
      org.opencontainers.image.licenses="MIT"
CMD ["/run.sh"]
