"""
Export the merged fine-tuned model to GGUF (q4_k_m) for llama.cpp / Ollama.

This is the self-hostable fallback path: if GPU hosting (HF Spaces/Modal)
isn't available or is too costly, the GGUF file runs CPU-only via
llama.cpp or Ollama, which is what backend/app/models/inference.py falls
back to when GEMMA_BACKEND=llamacpp.

Usage:
    python export_gguf.py --merged_dir outputs/merged --out_dir outputs/gguf
"""
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged_dir", default="outputs/merged")
    ap.add_argument("--out_dir", default="outputs/gguf")
    ap.add_argument("--quant", default="q4_k_m",
                     choices=["q4_k_m", "q5_k_m", "q8_0", "f16"])
    args = ap.parse_args()

    from unsloth import FastLanguageModel

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.merged_dir, max_seq_length=2048, dtype=None, load_in_4bit=False,
    )

    # Unsloth shells out to llama.cpp's convert + quantize under the hood.
    model.save_pretrained_gguf(str(out_dir), tokenizer, quantization_method=args.quant)

    print(f"GGUF model written to {out_dir} (quant={args.quant})")
    print(
        "To serve with Ollama:\n"
        f"  ollama create resume-jd-scorer -f {out_dir}/Modelfile\n"
        "  ollama run resume-jd-scorer\n"
        "Or with llama.cpp directly:\n"
        f"  ./llama-server -m {out_dir}/*.gguf -c 2048"
    )


if __name__ == "__main__":
    main()
