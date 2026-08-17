from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import ScoreRecord
from app.db.schemas import HistoryItem, HistoryPage

router = APIRouter(tags=["history"])


@router.get("/history", response_model=HistoryPage)
def get_history(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    total = db.scalar(select(func.count()).select_from(ScoreRecord)) or 0
    rows = (
        db.query(ScoreRecord)
        .order_by(ScoreRecord.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return HistoryPage(
        items=[HistoryItem.model_validate(r) for r in rows],
        total=total,
        page=page,
        page_size=page_size,
    )
