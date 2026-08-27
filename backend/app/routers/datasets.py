"""
Read-only view of the training dataset that `training/prepare_dataset.py`
produces. Powers the Datasets page. The .jsonl splits are git-ignored (they
regenerate from the script), so this endpoint degrades gracefully to
`available: false` when they aren't present.
"""
import json
import re
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter

router = APIRouter(tags=["datasets"])

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "processed"
SPLIT_FILES = ("train", "val", "test")
_SCORE_RE = re.compile(r"Score:\s*(\d{1,3})")


def _read_split(name: str) -> list[dict]:
    path = DATA_DIR / f"{name}.jsonl"
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@lru_cache
def _stats() -> dict:
    splits = {name: _read_split(name) for name in SPLIT_FILES}
    all_rows = [r for rows in splits.values() for r in rows]

    if not all_rows:
        return {
            "available": False,
            "source": "Kaggle Resume Dataset (snehaanbhawal/resume-dataset)",
            "labeler": "Gemini free tier (gemini-3.5-flash-lite), synthetic labels",
            "total": 0,
            "splits": [],
            "score_histogram": [],
            "score_mean": None,
            "instruction": None,
            "samples": [],
        }

    scores: list[int] = []
    for r in all_rows:
        m = _SCORE_RE.search(r.get("output", ""))
        if m:
            scores.append(max(0, min(100, int(m.group(1)))))

    buckets = [(i, i + 19) for i in range(0, 100, 20)]
    histogram = [
        {
            "bucket": f"{lo}-{hi if hi < 100 else 100}",
            "count": sum(1 for s in scores if lo <= s <= (hi if hi < 100 else 100)),
        }
        for lo, hi in buckets
    ]

    samples = [
        {"input": r["input"], "output": r["output"]}
        for r in splits["train"][:3]
    ]

    return {
        "available": True,
        "source": "Kaggle Resume Dataset (snehaanbhawal/resume-dataset)",
        "labeler": "Gemini free tier (gemini-3.5-flash-lite), synthetic labels",
        "total": len(all_rows),
        "splits": [{"name": name, "count": len(rows)} for name, rows in splits.items()],
        "score_histogram": histogram,
        "score_mean": round(sum(scores) / len(scores), 1) if scores else None,
        "instruction": all_rows[0].get("instruction"),
        "samples": samples,
    }


@router.get("/datasets/stats")
def dataset_stats():
    return _stats()
