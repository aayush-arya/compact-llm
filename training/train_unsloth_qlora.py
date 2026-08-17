"""
QLoRA fine-tune of Gemma-3 4B-it on the resume<->JD relevance dataset, via Unsloth.

Designed to run on a free Colab/Kaggle T4 (16GB). Run cells top to bottom in
a notebook, or run this as a script with `python train_unsloth_qlora.py`.

Env vars:
    WANDB_API_KEY   - required to log to Weights & Biases
    HF_TOKEN        - required to pull the gated Gemma-3 weights from the Hub

Outputs:
    outputs/adapter/        - LoRA adapter only (small, ~50-100MB)
    outputs/merged/         - adapter merged into base weights, fp16 (for export_gguf.py)
"""
import argparse
import os
from pathlib import Path

from datasets import load_dataset


def build_prompt(example: dict) -> dict:
    text = (
        f"<start_of_turn>user\n{example['instruction']}\n\n{example['input']}<end_of_turn>\n"
        f"<start_of_turn>model\n{example['output']}<end_of_turn>\n"
    )
    return {"text": text}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="unsloth/gemma-3-4b-it-bnb-4bit")
    ap.add_argument("--train_file", default="data/processed/train.jsonl")
    ap.add_argument("--val_file", default="data/processed/val.jsonl")
    ap.add_argument("--output_dir", default="outputs")
    ap.add_argument("--max_seq_length", type=int, default=2048)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--per_device_batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--wandb_project", default="resume-jd-relevance-scorer")
    ap.add_argument("--run_name", default="gemma3-4b-qlora-r16")
    args = ap.parse_args()

    from unsloth import FastLanguageModel  # imported late: patches transformers on import
    import torch
    from trl import SFTTrainer, SFTConfig

    if os.environ.get("WANDB_API_KEY"):
        os.environ["WANDB_PROJECT"] = args.wandb_project
        report_to = "wandb"
    else:
        print("[warn] WANDB_API_KEY not set - training will proceed without W&B logging.")
        report_to = "none"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        dtype=None,          # auto-detect (bf16 on Ampere+, fp16 otherwise)
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,   # 0 is optimized/faster in Unsloth
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    train_ds = load_dataset("json", data_files=args.train_file, split="train").map(build_prompt)
    val_ds = load_dataset("json", data_files=args.val_file, split="train").map(build_prompt)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        args=SFTConfig(
            per_device_train_batch_size=args.per_device_batch_size,
            per_device_eval_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.grad_accum,
            warmup_ratio=0.05,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=50,
            save_total_limit=2,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=3407,
            output_dir=f"{args.output_dir}/checkpoints",
            report_to=report_to,
            run_name=args.run_name,
        ),
    )

    trainer.train()

    adapter_dir = Path(args.output_dir) / "adapter"
    merged_dir = Path(args.output_dir) / "merged"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    merged_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"Adapter saved to {adapter_dir}")

    model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
    print(f"Merged fp16 model saved to {merged_dir}")


if __name__ == "__main__":
    main()
