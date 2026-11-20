# Image de l'API. Contexte de construction : la racine du dépôt.
#   docker build -f backend/Dockerfile -t ledger-agent-api .
# Le dossier data-snapshot/ doit contenir le snapshot extrait (voir data-snapshot.json).

FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13 /uv /usr/local/bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    LEDGER_AGENT_ROOT=/app \
    DATA_DIR=/app/data \
    PATH="/app/backend/.venv/bin:$PATH"

WORKDIR /app/backend

# Dépendances d'abord : cette couche reste en cache tant que uv.lock ne change pas
COPY backend/pyproject.toml backend/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY backend/src ./src
RUN uv sync --frozen --no-dev

# Modèle d'embedding intégré à l'image : aucun téléchargement au démarrage
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='/app/data/models')"

COPY evaluation /app/evaluation
COPY data-snapshot /app/data

# Utilisateur sans privilèges ; il doit pouvoir écrire le cache LLM et le verrou de Qdrant
RUN useradd --create-home --uid 1000 app && chown -R app:app /app/data
USER app

EXPOSE 8000
CMD ["uvicorn", "ledger_agent.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
