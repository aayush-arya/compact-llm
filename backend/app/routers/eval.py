import json
import subprocess
import sys
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.config import get_settings
from app.db.schemas import EvalRunStatus

router = APIRouter(tags=["eval"])

_REPO_ROOT = Path(__file__).resolve().parents[3]
_settings = get_settings()
BENCHMARK_JSON = Path(_settings.benchmark_json_path or _REPO_ROOT / "docs" / "benchmark_results.json")
TRAINING_DIR = Path(_settings.training_dir or _REPO_ROOT / "training")

_run_state = {"status": "idle", "detail": ""}


@router.get("/eval/benchmark")
def get_benchmark():
    if not BENCHMARK_JSON.exists():
        raise HTTPException(
            status_code=404,
            detail="No benchmark results yet -- run training/eval_base_vs_finetuned.py first.",
        )
    return json.loads(BENCHMARK_JSON.read_text())


def _run_eval_subprocess():
    global _run_state
    _run_state = {"status": "running", "detail": "eval_base_vs_finetuned.py started"}
    try:
        subprocess.run(
            [sys.executable, "eval_base_vs_finetuned.py"],
            cwd=str(TRAINING_DIR),
            check=True,
        )
        _run_state = {"status": "complete", "detail": "benchmark_results.json refreshed"}
    except subprocess.CalledProcessError as e:
        _run_state = {"status": "failed", "detail": str(e)}


@router.post("/eval/run", response_model=EvalRunStatus)
def run_eval(background_tasks: BackgroundTasks):
    """Stretch endpoint: kicks off the held-out test set re-evaluation in the
    background. Requires MODEL_BACKEND=transformers and a GPU -- this loads
    both the base model and the adapter, so it's not something to expose on
    a public demo without auth/rate limiting."""
    if _run_state["status"] == "running":
        return EvalRunStatus(**_run_state)
    background_tasks.add_task(_run_eval_subprocess)
    return EvalRunStatus(status="started", detail="running training/eval_base_vs_finetuned.py in background")


@router.get("/eval/run/status", response_model=EvalRunStatus)
def eval_run_status():
    return EvalRunStatus(**_run_state)
