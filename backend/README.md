---
title: CompactLLM Backend
emoji: 🎯
colorFrom: purple
colorTo: green
sdk: docker
app_port: 8000
pinned: false
---

# Backend — CompactLLM

FastAPI service for the resume/JD relevance scorer. Loads the base model +
LoRA adapter once at startup and exposes both behind one API.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/health` | liveness + which model backend is active |
| `POST` | `/score` | SSE stream of the fine-tuned model's score + rationale |
| `POST` | `/compare` | base vs fine-tuned for the same (resume, JD) pair |
| `GET`  | `/history` | paginated past requests |
| `GET`  | `/eval/benchmark` | held-out test-set results (404 until eval runs) |
| `GET`  | `/datasets/stats` | training-data composition (degrades if splits absent) |

Interactive docs at `/docs`.

## Run locally

```bash
python -m venv .venv && . .venv/Scripts/activate   # or .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Defaults to `MODEL_BACKEND=mock` (a deterministic heuristic scorer, no weights
needed). See [`.env.example`](.env.example) for `transformers` (GPU) and
`ollama` (GGUF) backends and all config.

## Docker

```bash
docker build -t compactllm-backend .
docker run -p 8000:8000 compactllm-backend
```

## Deploy to Hugging Face Spaces (Docker SDK)

The frontmatter above is what a Space needs. From the repo root, push this
folder to a Space:

```bash
# one-time: create a Docker Space at huggingface.co/new-space, then
git remote add space https://huggingface.co/spaces/<user>/compactllm-backend
git subtree push --prefix backend space main
```

Set Space secrets/variables as needed:

- `MODEL_BACKEND` — `mock` for a weights-free demo, else `transformers` / `ollama`
- `CORS_ORIGINS` — `["https://<your-vercel-app>.vercel.app"]`
- `DATABASE_URL` — a persistent Postgres URL if you want history to survive restarts
- `BENCHMARK_JSON_PATH` — path to `benchmark_results.json` if you bundle one
