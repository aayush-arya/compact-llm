"""
Build the resume <-> job-description relevance dataset.

Pipeline:
  1. Read real resumes from the Kaggle "Resume Dataset" CSV
     (https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset).
  2. For each resume, ask Gemini to write five job descriptions spanning the
     fit spectrum -- from a near-ideal match down to a clear mismatch -- so the
     label distribution covers the whole 0-100 range instead of clustering at
     the extremes (the failure mode of an earlier "matching vs mismatched"
     two-way design).
  3. In the same call, Gemini scores the (resume, JD) pair 0-100 with a short
     rationale, judging only on evidence in the resume.
  4. Emit instruction-tuning examples and split 80/10/10, stratified by score
     band so train / val / test each span the full range.

This is a synthetic-label dataset. Documented explicitly (see README) as a
limitation: labels reflect Gemini's judgment, not verified hiring outcomes.
Good enough to prove a fine-tune can specialize a small model toward this
scoring behavior; not a substitute for real recruiter labels.

Usage:
    # put GEMINI_API_KEY=... in training/.env (git-ignored), or export it
    python prepare_dataset.py --input data/raw/Resume.csv --resumes 120
    # -> data/processed/{train,val,test}.jsonl  (~600 examples: 5 per resume)

Resumable: progress is checkpointed after every example, so a killed or
rate-limited run just needs to be restarted with the same arguments.
"""
import argparse
import collections
import concurrent.futures
import json
import os
import random
import re
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from google import genai
from google.genai.errors import APIError
from tqdm import tqdm

try:  # let the key live in training/.env instead of the shell environment
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).with_name(".env"))
except ImportError:
    pass

# gemini-3.6-flash's free tier is capped at 20 requests/day -- unusable here.
# gemini-3.5-flash-lite's free tier allows 15 requests/minute AND 500
# requests/day. One combined call per example (JD generation + grading) keeps
# each example at a single request, so the daily cap buys ~500 examples/day.
MODEL = "gemini-3.5-flash-lite"
MAX_REQUESTS_PER_MINUTE = 14

# Five job-description "distances" from the resume. The grader still scores each
# pair independently on evidence (see COMBINED_PROMPT); the mode only steers how
# far the generated JD sits from the candidate's actual background, which is
# what spreads the score distribution across the range.
MODES: dict[str, str] = {
    "strong": (
        "a role this candidate is an excellent fit for: the same function and "
        "seniority as the resume, where the resume already demonstrates every "
        "major requirement."
    ),
    "solid": (
        "a role in the same field and seniority as the resume, but which asks "
        "for one or two specific skills, tools, or domains the resume does not "
        "clearly show -- a good but imperfect fit."
    ),
    "partial": (
        "a role adjacent to the resume -- a neighbouring specialisation, or one "
        "notch of seniority away -- so only part of the candidate's experience "
        "transfers."
    ),
    "weak": (
        "a role that shares only a broad industry or a few transferable skills "
        "with the resume: wrong function or wrong domain, the kind of posting a "
        "screener would rate a weak match."
    ),
    "mismatch": (
        "a role in a clearly different field that needs skills this candidate "
        "does not have, with little transferable overlap."
    ),
}

COMBINED_PROMPT = """You are building a training dataset for a resume-screening model.
Do two things in a single response.

Step 1 -- job description. Given the resume below, write a realistic job
description (150-300 words: a title line, then responsibilities and
requirements). Aim for this fit level relative to the candidate: {mode_guidance}
Do not mention the resume or the candidate. Write it the way a real posting
would read, not as an echo of the resume text.

Step 2 -- score. As an experienced technical recruiter, score how well the
resume matches the job description you just wrote (0-100). Judge ONLY on
evidence in the resume: relevant skills, years of relevant experience, domain
overlap, seniority match. Score what is actually on the page, not how the JD
was commissioned. Do not reward keyword stuffing.

Resume:
{resume}

Respond in EXACTLY this format, nothing else, no markdown:
Job Description: <the job description text>
Score: <integer 0-100>
Rationale: <exactly 2 sentences explaining the score, citing specific
resume/JD evidence>
"""

INSTRUCTION = (
    "Score how well this resume matches this job description (0-100) and "
    "explain why in 2 sentences."
)


def _seconds_until_pacific_midnight() -> float:
    """Google's free-tier daily quota resets at midnight Pacific time."""
    now = datetime.now(ZoneInfo("America/Los_Angeles"))
    reset = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return (reset - now).total_seconds()


class RateLimiter:
    """Caps calls to a rolling per-minute window, shared across worker threads."""

    def __init__(self, max_per_minute: int):
        self.max_per_minute = max_per_minute
        self.timestamps: collections.deque = collections.deque()
        self.lock = threading.Lock()

    def acquire(self):
        while True:
            with self.lock:
                now = time.monotonic()
                while self.timestamps and now - self.timestamps[0] > 60:
                    self.timestamps.popleft()
                if len(self.timestamps) < self.max_per_minute:
                    self.timestamps.append(now)
                    return
                sleep_for = 60 - (now - self.timestamps[0]) + 0.1
            time.sleep(sleep_for)


rate_limiter = RateLimiter(MAX_REQUESTS_PER_MINUTE)


def call_gemini(client: genai.Client, prompt: str) -> str:
    backoff = 5.0
    attempt = 0
    while True:
        rate_limiter.acquire()
        try:
            resp = client.models.generate_content(model=MODEL, contents=prompt)
            return (resp.text or "").strip()
        except APIError as e:
            if "PerDay" in str(e):
                wait_s = _seconds_until_pacific_midnight() + 300
                print(
                    f"  [info] daily free-tier quota hit -- sleeping ~{wait_s / 3600:.1f}h until reset...",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(wait_s)
                continue  # doesn't count against the attempt budget below
            if e.code in (429, 500, 503) and attempt < 6:
                time.sleep(backoff)
                backoff = min(backoff * 1.8, 60)
                attempt += 1
                continue
            raise
        except OSError:
            # transient network/DNS hiccup (e.g. brief connectivity loss)
            if attempt < 6:
                time.sleep(backoff)
                backoff = min(backoff * 1.8, 60)
                attempt += 1
                continue
            raise


def parse_combined(text: str) -> tuple[str, int, str] | None:
    jd_m = re.search(r"Job Description:\s*(.+?)\s*Score:", text, re.DOTALL)
    score_m = re.search(r"Score:\s*(\d{1,3})", text)
    rationale_m = re.search(r"Rationale:\s*(.+)", text, re.DOTALL)
    if not jd_m or not score_m or not rationale_m:
        return None
    jd = jd_m.group(1).strip()
    score = max(0, min(100, int(score_m.group(1))))
    rationale = " ".join(rationale_m.group(1).strip().split())
    return jd, score, rationale


def process_one(client: genai.Client, resume: str, mode: str) -> dict | None:
    raw = call_gemini(client, COMBINED_PROMPT.format(mode_guidance=MODES[mode], resume=resume))
    parsed = parse_combined(raw)
    if parsed is None:
        return None
    jd, score, rationale = parsed
    return {
        "instruction": INSTRUCTION,
        "input": f"Resume: {resume}\n\nJob Description: {jd}",
        "output": f"Score: {score}\nRationale: {rationale}",
        "meta": {"mode": mode, "score": score},
    }


def load_resumes(csv_path: Path, n: int, seed: int) -> list[str]:
    df = pd.read_csv(csv_path)
    text_col = next(
        (c for c in df.columns if c.lower() in ("resume_str", "resume", "resume_text")), None
    )
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


def stratified_split(rows: list[dict], seed: int) -> dict[str, list[dict]]:
    """Split 80/10/10 within each 20-point score band, so train / val / test all
    span the range instead of a random slice landing lopsided."""
    rng = random.Random(seed)
    buckets: dict[int, list[dict]] = collections.defaultdict(list)
    for r in rows:
        band = min(4, r["meta"]["score"] // 20)  # 0-19, 20-39, 40-59, 60-79, 80-100
        buckets[band].append(r)

    train: list[dict] = []
    val: list[dict] = []
    test: list[dict] = []
    for band in sorted(buckets):
        items = buckets[band]
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        train += items[:n_train]
        val += items[n_train : n_train + n_val]
        test += items[n_train + n_val :]

    for split in (train, val, test):
        rng.shuffle(split)
    return {"train": train, "val": val, "test": test}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input", type=Path, default=Path("data/raw/Resume.csv"), help="Kaggle resume CSV path"
    )
    ap.add_argument(
        "--resumes",
        type=int,
        default=120,
        help="number of resumes to sample (each produces 5 examples, one per fit mode)",
    )
    ap.add_argument("--out_dir", type=Path, default=Path("data/processed"))
    ap.add_argument(
        "--checkpoint", type=Path, default=Path("data/raw/dataset_checkpoint.jsonl")
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=4,
        help="concurrency is capped by the global rate limiter regardless of this value",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit(
            "Set GEMINI_API_KEY before running this script -- either export it or put\n"
            "  GEMINI_API_KEY=...\n"
            "in training/.env (git-ignored). Free tier: https://aistudio.google.com/apikey"
        )
    if not args.input.exists():
        sys.exit(
            f"Resume CSV not found at {args.input}.\n"
            "Download it first, e.g.:\n"
            "  kaggle datasets download -d snehaanbhawal/resume-dataset -p data/raw --unzip"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    resumes = load_resumes(args.input, args.resumes, args.seed)
    jobs = [(r, mode) for r in resumes for mode in MODES]

    done = load_checkpoint(args.checkpoint)
    done_keys = {(d["input"][:80], d["meta"]["mode"]) for d in done}
    print(f"Resuming: {len(done)} examples already generated.")

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    ckpt_f = open(args.checkpoint, "a", encoding="utf-8")

    def task(job):
        resume, mode = job
        key = (f"Resume: {resume}"[:80], mode)
        if key in done_keys:
            return None
        try:
            return process_one(client, resume, mode)
        except Exception as e:  # noqa: BLE001 - one bad example shouldn't kill the run
            print(f"  [warn] failed one example ({mode}): {e}", file=sys.stderr)
            return None

    remaining = [j for j in jobs if (f"Resume: {j[0]}"[:80], j[1]) not in done_keys]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in tqdm(pool.map(task, remaining), total=len(remaining), desc="labeling"):
            if result is not None:
                ckpt_f.write(json.dumps(result) + "\n")
                ckpt_f.flush()
    ckpt_f.close()

    all_rows = load_checkpoint(args.checkpoint)
    splits = stratified_split(all_rows, args.seed)
    for name, rows in splits.items():
        out_path = args.out_dir / f"{name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for row in rows:
                clean = {k: row[k] for k in ("instruction", "input", "output")}
                f.write(json.dumps(clean) + "\n")
        print(f"{name}: {len(rows)} examples -> {out_path}")

    scores = [r["meta"]["score"] for r in all_rows]
    by_band = collections.Counter(min(4, s // 20) for s in scores)
    band_labels = ["0-19", "20-39", "40-59", "60-79", "80-100"]
    dist = "  ".join(f"{band_labels[b]}:{by_band.get(b, 0)}" for b in range(5))
    print(
        f"\nTotal: {len(all_rows)} examples. "
        f"Score min={min(scores)} max={max(scores)} mean={sum(scores) / len(scores):.1f}"
    )
    print(f"Distribution by band: {dist}")
    print("IMPORTANT: the held-out test.jsonl must not be touched again until eval time.")


if __name__ == "__main__":
    main()
