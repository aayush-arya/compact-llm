"""
Build the resume <-> job-description relevance dataset.

Pipeline:
  1. Read real resumes from the Kaggle "Resume Dataset" CSV
     (https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset).
  2. For each resume, use Claude as a synthetic-JD generator to produce
     one "matching" JD (same field, plausible fit) and one "mismatched"
     JD (different field / seniority), so the label distribution isn't
     bunched at the top end.
  3. Use Claude again as a labeling oracle: given (resume, JD), produce
     a 0-100 relevance score + 2-sentence rationale.
  4. Emit instruction-tuning examples and split 80/10/10 train/val/test.

This is a synthetic-label dataset. Documented explicitly (see README) as
a limitation: labels reflect Claude's judgment, not verified hiring
outcomes. Good enough to prove a fine-tune can specialize a small model
toward this scoring behavior; not a substitute for real recruiter labels.

Usage:
    export ANTHROPIC_API_KEY=...
    python prepare_dataset.py --input data/raw/Resume.csv --n 1500

Resumable: progress is checkpointed to --checkpoint after every example,
so a killed/rate-limited run can just be restarted.
"""
import argparse
import concurrent.futures
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import pandas as pd
from anthropic import Anthropic, APIStatusError
from tqdm import tqdm

GENERATOR_MODEL = "claude-sonnet-5"
GRADER_MODEL = "claude-sonnet-5"

JD_GEN_PROMPT = """You are helping build a training dataset for a resume-screening model.

Given the resume below, write a realistic job description.
Mode: {mode}
- "matching": a job this candidate would be a strong-to-decent fit for (same
  general field/seniority as the resume, but don't make it a perfect echo of
  the resume text -- phrase it the way a real job posting would be written).
- "mismatched": a job in a different field, seniority level, or requiring
  skills this candidate mostly lacks, so a real screener would rate it a
  weak match.

Write 150-300 words: a title line, then responsibilities and requirements.
Do not mention the resume or the candidate. Output ONLY the job description
text, no preamble, no markdown headers.

Resume:
{resume}
"""

SCORE_PROMPT = """You are an experienced technical recruiter scoring resume-to-job-description
fit. Score strictly on evidence in the resume: skills, years of relevant
experience, domain overlap, seniority match. Do not reward keyword stuffing.

Resume:
{resume}

Job Description:
{jd}

Respond in EXACTLY this format, nothing else:
Score: <integer 0-100>
Rationale: <exactly 2 sentences explaining the score, citing specific
resume/JD evidence>
"""

INSTRUCTION = (
    "Score how well this resume matches this job description (0-100) and "
    "explain why in 2 sentences."
)


def call_claude(client: Anthropic, model: str, prompt: str, max_tokens: int = 500) -> str:
    backoff = 2.0
    for attempt in range(6):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text.strip()
        except APIStatusError as e:
            if e.status_code in (429, 529, 500, 502, 503) and attempt < 5:
                time.sleep(backoff)
                backoff *= 1.8
                continue
            raise
    raise RuntimeError("exhausted retries calling Claude")


def parse_score(text: str) -> tuple[int, str] | None:
    m = re.search(r"Score:\s*(\d{1,3})", text)
    r = re.search(r"Rationale:\s*(.+)", text, re.DOTALL)
    if not m or not r:
        return None
    score = max(0, min(100, int(m.group(1))))
    rationale = " ".join(r.group(1).strip().split())
    return score, rationale


def process_one(client: Anthropic, resume: str, mode: str) -> dict | None:
    jd = call_claude(client, GENERATOR_MODEL, JD_GEN_PROMPT.format(mode=mode, resume=resume))
    raw_score = call_claude(client, GRADER_MODEL, SCORE_PROMPT.format(resume=resume, jd=jd))
    parsed = parse_score(raw_score)
    if parsed is None:
        return None
    score, rationale = parsed
    return {
        "instruction": INSTRUCTION,
        "input": f"Resume: {resume}\n\nJob Description: {jd}",
        "output": f"Score: {score}\nRationale: {rationale}",
        "meta": {"mode": mode, "score": score},
    }


def load_resumes(csv_path: Path, n: int, seed: int) -> list[str]:
    df = pd.read_csv(csv_path)
    text_col = next((c for c in df.columns if c.lower() in ("resume_str", "resume", "resume_text")), None)
    if text_col is None:
        raise ValueError(f"Couldn't find a resume text column in {list(df.columns)}")
    texts = df[text_col].dropna().astype(str)
    texts = texts[texts.str.len() > 200]  # drop near-empty rows
    texts = texts.drop_duplicates()
    text_list = texts.tolist()
    random.Random(seed).shuffle(text_list)
    return text_list[:n]


def load_checkpoint(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("data/raw/Resume.csv"),
                     help="Kaggle resume CSV path")
    ap.add_argument("--n", type=int, default=900,
                     help="number of resumes to sample (each produces 2 examples: matching + mismatched)")
    ap.add_argument("--out_dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--checkpoint", type=Path, default=Path("data/raw/generation_checkpoint.jsonl"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY before running this script.")
    if not args.input.exists():
        sys.exit(
            f"Resume CSV not found at {args.input}.\n"
            "Download it first, e.g.:\n"
            "  kaggle datasets download -d snehaanbhawal/resume-dataset -p data/raw --unzip"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    resumes = load_resumes(args.input, args.n, args.seed)
    jobs = [(r, mode) for r in resumes for mode in ("matching", "mismatched")]

    done = load_checkpoint(args.checkpoint)
    done_keys = {(d["input"][:80], d["meta"]["mode"]) for d in done}
    print(f"Resuming: {len(done)} examples already generated.")

    client = Anthropic()
    ckpt_f = open(args.checkpoint, "a", encoding="utf-8")

    def task(job):
        resume, mode = job
        key = (f"Resume: {resume}"[:80], mode)
        if key in done_keys:
            return None
        try:
            return process_one(client, resume, mode)
        except Exception as e:
            print(f"  [warn] failed one example: {e}", file=sys.stderr)
            return None

    remaining = [j for j in jobs if (f"Resume: {j[0]}"[:80], j[1]) not in done_keys]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in tqdm(pool.map(task, remaining), total=len(remaining), desc="labeling"):
            if result is not None:
                ckpt_f.write(json.dumps(result) + "\n")
                ckpt_f.flush()
    ckpt_f.close()

    all_rows = load_checkpoint(args.checkpoint)
    random.Random(args.seed).shuffle(all_rows)
    n = len(all_rows)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    splits = {
        "train": all_rows[:n_train],
        "val": all_rows[n_train:n_train + n_val],
        "test": all_rows[n_train + n_val:],
    }
    for name, rows in splits.items():
        out_path = args.out_dir / f"{name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for row in rows:
                clean = {k: row[k] for k in ("instruction", "input", "output")}
                f.write(json.dumps(clean) + "\n")
        print(f"{name}: {len(rows)} examples -> {out_path}")

    scores = [r["meta"]["score"] for r in all_rows]
    print(f"\nTotal: {n} examples. Score distribution: "
          f"min={min(scores)} max={max(scores)} mean={sum(scores)/n:.1f}")
    print("IMPORTANT: the held-out test.jsonl must not be touched again until eval time.")


if __name__ == "__main__":
    main()
