"""Application FastAPI.

Lancement local :
    uv run uvicorn ledger_agent.api.main:app --reload --port 8000
"""

import json
import logging
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from ledger_agent.api.schemas import AnswerResponse, ChatRequest, Company, Health
from ledger_agent.api.service import AgentService, LocalAgentService
from ledger_agent.config import REPO_ROOT, get_settings

logger = logging.getLogger(__name__)

REPORTS_DIR = REPO_ROOT / "evaluation" / "reports"

ServiceFactory = Callable[[], AgentService]


def sse(event: dict[str, Any]) -> str:
    """Formate un événement Server-Sent Events : son type, puis ses données JSON."""
    payload = json.dumps(event, ensure_ascii=False, default=str)
    return f"event: {event.get('type', 'message')}\ndata: {payload}\n\n"


def create_app(service_factory: ServiceFactory | None = None) -> FastAPI:
    settings = get_settings()
    factory = service_factory or (lambda: LocalAgentService(settings))

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # L'agent (modèles d'embedding, index, base de faits) est chargé une seule fois
        app.state.service = factory()
        try:
            yield
        finally:
            app.state.service.close()

    app = FastAPI(
        title="ledger-agent",
        description="Analyse financière sur les rapports SEC, avec chiffres traçables.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    def service(request: Request) -> AgentService:
        svc: AgentService = request.app.state.service
        return svc

    @app.get("/health", response_model=Health)
    def health(request: Request) -> Health:
        svc = service(request)
        return Health(status="ok", ready=svc.ready, config=svc.config_name)

    @app.get("/companies", response_model=list[Company])
    def companies(request: Request) -> list[Company]:
        return [Company(**c) for c in service(request).companies()]

    @app.get("/benchmark")
    def benchmark(split: Literal["dev", "test", "dev_rapide"] = "dev") -> dict[str, Any]:
        """Dernier tableau d'ablation produit par `poe eval-report`."""
        path = REPORTS_DIR / f"ablation-{split}.json"
        if not path.exists():
            raise HTTPException(status_code=404, detail="Aucun rapport d'ablation pour ce jeu")
        report: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return report

    @app.post("/chat")
    def chat(body: ChatRequest, request: Request) -> StreamingResponse:
        """Diffuse les étapes de l'agent au fil de l'eau, puis la réponse finale (SSE)."""
        events: Iterator[dict[str, Any]] = service(request).stream(body.question)
        return StreamingResponse(
            (sse(e) for e in events),
            media_type="text/event-stream",
            # Empêche les proxys de retenir le flux jusqu'à la fin
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/ask", response_model=AnswerResponse)
    def ask(body: ChatRequest, request: Request) -> AnswerResponse:
        """Même traitement que /chat, mais renvoie une seule réponse JSON à la fin."""
        steps: list[dict[str, Any]] = []
        for event in service(request).stream(body.question):
            if event["type"] == "error":
                status = 429 if event.get("code") == "quota" else 502
                raise HTTPException(status_code=status, detail=event["message"])
            if event["type"] == "answer":
                return AnswerResponse(answer=event, steps=steps)
            steps.append(event)
        raise HTTPException(status_code=500, detail="L'agent n'a pas produit de réponse")

    return app


app = create_app()
