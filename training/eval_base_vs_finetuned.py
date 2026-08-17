"""
Phase 4 evaluation: base zero-shot vs base few-shot vs fine-tuned, on the
held-out test split. This is the script that produces docs/benchmark_table.md
and the JSON the /eval/benchmark API endpoint serves.

Metrics per approach:
    - Pearson & Spearman correlation (predicted score vs human/Claude label)
    - MAE (mean absolute error)
    - rationale quality, graded 1-5 by Claude as an LLM judge
    - mean latency per request (seconds)

Run AFTER training finishes, on data/processed/test.jsonl only (never used
during training or hyperparameter selection).

Usage:
    export ANTHROPIC_API_KEY=...   # for the LLM-judge rationale grading
    python eval_base_vs_finetuned.py \
        --test_file data/processed/test.jsonl \
        --adapter_dir outputs/adapter \
        --base_model unsloth/gemma-3-4b-it-bnb-4bit
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

FEW_SHOT_EXAMPLES = """Example 1:
Resume: 5 years as a backend engineer, Python/Django, led a team of 3, built payment
processing systems handling $2M/day in transactions.
Job Description: Senior Backend Engineer, fintech startup. Requires 4+ years Python,
experience with payment systems, team leadership a plus.
Score: 92
Rationale: Direct domain overlap in payments and matching seniority with leadership
experience the JD calls out as a plus. Years of experience exceed the stated minimum.

Example 2:
Resume: Recent bootcamp grad, built 2 React portfolio projects, no professional
experience.
Job Description: Staff iOS Engineer, 8+ years Swift/Objective-C, must have shipped
3+ App Store apps at scale.
Score: 8
Rationale: No iOS or Swift experience of any kind and zero professional experience
against an 8-year seniority bar. Portfolio projects are unrelated web frontend work.

"""

ZERO_SHOT_PROMPT = """Score how well this resume matches this job description (0-100) and explain why in 2 sentences.

{input}

Respond in EXACTLY this format:
Score: <integer 0-100>
Rationale: <2 sentences>"""

FEW_SHOT_PROMPT = FEW_SHOT_EXAMPLES + """Now score this one:

{input}

Respond in EXACTLY this format:
Score: <integer 0-100>
Rationale: <2 sentences>"""

JUDGE_PROMPT = """Rate the quality of this rationale for a resume/JD relevance score, on a
1-5 scale (5 = specific, evidence-based, correctly justifies the score;
1 = generic, vague, or contradicts the score). Respond with ONLY the integer.

Score given: {score}
Rationale: {rationale}
"""


def parse_output(text: str) -> tuple[int | None, str]:
    m = re.search(r"Score:\s*(\d{1,3})", text)
    r = re.search(r"Rationale:\s*(.+)", text, re.DOTALL)
    score = max(0, min(100, int(m.group(1)))) if m else None
    rationale = " ".join(r.group(1).strip().split()) if r else text.strip()[:300]
    return score, rationale


def run_hf_generation(model, tokenizer, prompt: str, max_new_tokens: int = 150) -> tuple[str, float]:
    import torch
    inputs = tokenizer([f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"],
                        return_tensors="pt").to(model.device)
    start = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True,
                              do_sample=False, temperature=1.0)
    latency = time.perf_counter() - start
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text, latency


def evaluate_approach(name: str, gen_fn, test_rows: list[dict], judge_client, judge_model: str) -> dict:
    preds, labels, latencies, judge_scores = [], [], [], []
    for row in tqdm(test_rows, desc=name):
        gold_score, _ = parse_output(row["output"])
        pred_text, latency = gen_fn(row)
        pred_score, pred_rationale = parse_output(pred_text)
        if pred_score is None or gold_score is None:
            continue
        preds.append(pred_score)
        labels.append(gold_score)
        latencies.append(latency)

        judge_resp = judge_client.messages.create(
            model=judge_model, max_tokens=5,
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(score=pred_score, rationale=pred_rationale)}],
        )
        m = re.search(r"\d", judge_resp.content[0].text)
        judge_scores.append(int(m.group()) if m else 3)

    preds_a, labels_a = np.array(preds), np.array(labels)
    pearson, _ = pearsonr(preds_a, labels_a) if len(preds_a) > 1 else (float("nan"), None)
    spearman, _ = spearmanr(preds_a, labels_a) if len(preds_a) > 1 else (float("nan"), None)
    mae = float(np.mean(np.abs(preds_a - labels_a)))

    return {
        "approach": name,
        "n": len(preds),
        "pearson": round(float(pearson), 3),
        "spearman": round(float(spearman), 3),
        "mae": round(mae, 2),
        "rationale_quality_1_5": round(float(np.mean(judge_scores)), 2),
        "mean_latency_sec": round(float(np.mean(latencies)), 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_file", default="data/processed/test.jsonl")
    ap.add_argument("--base_model", default="unsloth/gemma-3-4b-it-bnb-4bit")
    ap.add_argument("--adapter_dir", default="outputs/adapter")
    ap.add_argument("--judge_model", default="claude-sonnet-5")
    ap.add_argument("--max_examples", type=int, default=None)
    ap.add_argument("--out_json", default="../docs/benchmark_results.json")
    ap.add_argument("--out_md", default="../docs/benchmark_table.md")
    args = ap.parse_args()

    from unsloth import FastLanguageModel
    from anthropic import Anthropic

    test_rows = [json.loads(l) for l in open(args.test_file, encoding="utf-8")]
    if args.max_examples:
        test_rows = test_rows[: args.max_examples]

    judge_client = Anthropic()

    print("Loading base model...")
    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model, max_seq_length=2048, dtype=None, load_in_4bit=True,
    )
    FastLanguageModel.for_inference(base_model)

    results = []
    results.append(evaluate_approach(
        "base_zero_shot",
        lambda row: run_hf_generation(base_model, tokenizer, ZERO_SHOT_PROMPT.format(input=row["input"])),
        test_rows, judge_client, args.judge_model,
    ))
    results.append(evaluate_approach(
        "base_few_shot",
        lambda row: run_hf_generation(base_model, tokenizer, FEW_SHOT_PROMPT.format(input=row["input"])),
        test_rows, judge_client, args.judge_model,
    ))

    print("Loading fine-tuned adapter...")
    ft_model, ft_tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter_dir, max_seq_length=2048, dtype=None, load_in_4bit=True,
    )
    FastLanguageModel.for_inference(ft_model)
    results.append(evaluate_approach(
        "fine_tuned",
        lambda row: run_hf_generation(ft_model, ft_tokenizer, f"{row['instruction']}\n\n{row['input']}"),
        test_rows, judge_client, args.judge_model,
    ))

    out_json_path = Path(args.out_json)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_json_path}")

    header = "| Approach | N | Pearson r | Spearman ρ | MAE | Rationale quality (1-5) | Mean latency (s) |"
    sep = "|---|---|---|---|---|---|---|"
    rows = [header, sep]
    for r in results:
        rows.append(
            f"| {r['approach']} | {r['n']} | {r['pearson']} | {r['spearman']} | "
            f"{r['mae']} | {r['rationale_quality_1_5']} | {r['mean_latency_sec']} |"
        )
    Path(args.out_md).write_text("\n".join(rows) + "\n")
    print(f"Wrote {args.out_md}")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
