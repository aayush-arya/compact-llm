from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.db.database import init_db
from app.models.inference import get_engine
from app.routers import compare, eval, history, score


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    get_engine()  # load model weights once at startup, not per-request
    yield


app = FastAPI(
    title="Resume/JD Relevance Scorer",
    description="Fine-tuned Gemma-3 4B (QLoRA via Unsloth) vs base model comparison API",
    version="0.1.0",
    lifespan=lifespan,
)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(score.router)
app.include_router(compare.router)
app.include_router(history.router)
app.include_router(eval.router)


@app.get("/health")
def health():
    return {"status": "ok", "model_backend": settings.model_backend}
