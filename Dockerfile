# VPN/Proxy IP Classifier API.
#   docker build -t vpn-proxy-api .
#   docker run -p 8000:8000 vpn-proxy-api        then open http://localhost:8000/docs
#
# Stage 1 installs the runtime packages into a virtual environment; stage 2 copies only that
# environment, the source code and the trained model, and runs as a non-root user.

FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
COPY requirements-api.txt .
RUN /opt/venv/bin/pip install -r requirements-api.txt


FROM python:3.12-slim
# The commit the image was built from; shown by GET /health as "build" so you can see what is deployed.
ARG GIT_SHA=unknown
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NUMBA_CACHE_DIR=/tmp/numba \
    PORT=8000 \
    GIT_SHA=$GIT_SHA
LABEL org.opencontainers.image.source="https://github.com/uzair-khiiba/VPN-Classifier-MLOps" \
      org.opencontainers.image.description="VPN/Proxy IP classifier (FastAPI)" \
      org.opencontainers.image.revision=$GIT_SHA

RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv

# Source code, the trained model with its lookup data, request examples and the saved SHAP ranking.
COPY src/ src/
COPY examples/ examples/
COPY models/model.joblib models/model_meta.json models/feature_importance.json \
     models/ip_lookup.parquet models/feature_reference.json models/
COPY reports/figures/shap/ reports/figures/shap/
RUN chown -R app:app /app
USER app

EXPOSE 8000
# Healthy only when /health answers and reports a loaded model ("ok", not "degraded").
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import json, os, urllib.request as u; \
r = json.load(u.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=4)); \
exit(0 if r['status'] == 'ok' else 1)"

# PORT is 8000 by default; hosting platforms such as Render set their own.
CMD ["sh", "-c", "exec uvicorn src.app:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
