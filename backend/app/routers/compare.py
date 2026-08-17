from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import ScoreRecord
from app.db.schemas import CompareResponse, ScoreRequest, ScoreResult
from app.models.inference import get_engine

router = APIRouter(tags=["compare"])


@router.post("/compare", response_model=CompareResponse)
def compare(req: ScoreRequest, db: Session = Depends(get_db)):
    """The centerpiece endpoint: runs the same (resume, JD) pair through both
    the base model and the fine-tuned adapter, off the same loaded weights,
    and returns both so the frontend can show the delta side by side."""
    engine = get_engine()

    base = engine.generate(req.resume, req.job_description, use_adapter=False)
    finetuned = engine.generate(req.resume, req.job_description, use_adapter=True)

    record = ScoreRecord(
        resume_text=req.resume,
        jd_text=req.job_description,
        finetuned_score=finetuned.score,
        finetuned_rationale=finetuned.rationale,
        finetuned_latency_ms=finetuned.latency_ms,
        base_score=base.score,
        base_rationale=base.rationale,
        base_latency_ms=base.latency_ms,
        is_comparison=True,
    )
    db.add(record)
    db.commit()

    return CompareResponse(
        base=ScoreResult(score=base.score, rationale=base.rationale, latency_ms=base.latency_ms),
        finetuned=ScoreResult(score=finetuned.score, rationale=finetuned.rationale, latency_ms=finetuned.latency_ms),
        score_delta=finetuned.score - base.score,
    )
