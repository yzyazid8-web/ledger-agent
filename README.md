# ledger-agent

Agentic RAG over SEC filings (10-K) that answers financial questions with **fully traceable numbers**:
every figure in an answer comes from an XBRL fact, a cited passage or a logged calculation, or the
agent abstains.

> Work in progress.

## Repository layout

| Folder | Content |
|---|---|
| `backend/` | Python package: ingestion, retrieval, tools, LangGraph agent, FastAPI |
| `evaluation/` | FinanceBench benchmark and ablation study |
| `frontend/` | Next.js interface |
| `infra/` | Azure Container Apps (Bicep) |
| `docs/` | Architecture and decision records |

## Local setup

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), Node.js 20+.

```bash
cp .env.example .env        # then fill in the values
cd backend
uv sync
uv run poe check            # lint, format check, type check, tests
```
