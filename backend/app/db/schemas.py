import datetime

from pydantic import BaseModel, Field


class ScoreRequest(BaseModel):
    resume: str = Field(min_length=1)
    job_description: str = Field(min_length=1)


class ScoreResult(BaseModel):
    score: int
    rationale: str
    latency_ms: float


class ScoreResponse(ScoreResult):
    pass


class CompareResponse(BaseModel):
    base: ScoreResult
    finetuned: ScoreResult
    score_delta: int


class HistoryItem(BaseModel):
    id: int
    resume_text: str
    jd_text: str
    finetuned_score: int
    finetuned_rationale: str
    base_score: int | None
    is_comparison: bool
    created_at: datetime.datetime

    model_config = {"from_attributes": True}


class HistoryPage(BaseModel):
    items: list[HistoryItem]
    total: int
    page: int
    page_size: int


class BenchmarkRow(BaseModel):
    approach: str
    n: int
    pearson: float
    spearman: float
    mae: float
    rationale_quality_1_5: float
    mean_latency_sec: float


class EvalRunStatus(BaseModel):
    status: str
    detail: str
