import json
import time

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import ScoreRecord
from app.db.schemas import ScoreRequest
from app.models.inference import get_engine, parse_output

router = APIRouter(tags=["score"])


@router.post("/score")
def score(req: ScoreRequest, db: Session = Depends(get_db)):
    """Streams the fine-tuned model's score + rationale via SSE as it's generated,
    so the frontend can render tokens live instead of showing a blocking spinner."""
    engine = get_engine()

    def event_stream():
        start = time.perf_counter()
        accumulated = ""
        for chunk in engine.generate_stream(req.resume, req.job_description, use_adapter=True):
            accumulated += chunk
            yield f"data: {json.dumps({'type': 'chunk', 'text': chunk})}\n\n"

        latency_ms = (time.perf_counter() - start) * 1000
        score_val, rationale = parse_output(accumulated)

        record = ScoreRecord(
            resume_text=req.resume,
            jd_text=req.job_description,
            finetuned_score=score_val,
            finetuned_rationale=rationale,
            finetuned_latency_ms=latency_ms,
            is_comparison=False,
        )
        db.add(record)
        db.commit()

        yield f"data: {json.dumps({'type': 'done', 'score': score_val, 'rationale': rationale, 'latency_ms': latency_ms})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
