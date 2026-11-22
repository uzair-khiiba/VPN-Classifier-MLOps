# Streamlit dashboard for the VPN/Proxy IP Classifier. It has no model: every prediction is an HTTP call to the API.
#   docker build -f ui.Dockerfile -t vpn-proxy-dashboard .
#   docker run -p 8501:8501 -e API_BASE_URL=https://vpn-proxy-api.onrender.com vpn-proxy-dashboard
#
# Files to leave out are listed in ui.Dockerfile.dockerignore, which Docker uses instead of .dockerignore.

FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
COPY requirements-dashboard.txt .
RUN /opt/venv/bin/pip install -r requirements-dashboard.txt


FROM python:3.12-slim
ARG GIT_SHA=unknown
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8501 \
    API_BASE_URL=https://vpn-proxy-api.onrender.com \
    GIT_SHA=$GIT_SHA
LABEL org.opencontainers.image.source="https://github.com/uzair-khiiba/VPN-Classifier-MLOps" \
      org.opencontainers.image.description="VPN/Proxy IP classifier dashboard (Streamlit)" \
      org.opencontainers.image.revision=$GIT_SHA

RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv

# The dashboard code and theme, and the saved results its Performance and Explainability pages show.
COPY ui/ ui/
COPY .streamlit/ .streamlit/
COPY models/model_meta.json models/feature_importance.json models/
COPY reports/experiments/results/ reports/experiments/results/
COPY reports/experiments/figures/ reports/experiments/figures/
COPY reports/figures/shap/ reports/figures/shap/
RUN chown -R app:app /app
USER app

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request as u; u.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/_stcore/health', timeout=4)"

# .streamlit/config.toml binds to 127.0.0.1 for local use; in a container it must listen on every interface.
CMD ["sh", "-c", "exec streamlit run ui/streamlit_app.py --server.address=0.0.0.0 --server.port=${PORT} --server.headless=true"]
