import datetime

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


class ScoreRecord(Base):
    __tablename__ = "score_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    resume_text: Mapped[str] = mapped_column(Text)
    jd_text: Mapped[str] = mapped_column(Text)

    finetuned_score: Mapped[int] = mapped_column(Integer)
    finetuned_rationale: Mapped[str] = mapped_column(Text)
    finetuned_latency_ms: Mapped[float] = mapped_column(Float)

    base_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    base_rationale: Mapped[str | None] = mapped_column(Text, nullable=True)
    base_latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)

    is_comparison: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc)
    )
