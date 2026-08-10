"""Export frozen GSM8K regression responses from a local checkpoint.

    uv run python -m eval.export_gsm8k_regression_sample \\
        --checkpoint outputs/qwen3.5-4b-mining \\
        --out eval/results/gsm8k_regression_sample.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eval.gsm8k_eval import format_gsm8k_prompt
from eval.regression_sample import REGRESSION_PROBLEMS_PATH, build_regression_sample, load_regression_problems


def export_gsm8k_regression_sample(
    checkpoint: Path,
    *,
    out_path: Path,
    problems_path: Path = REGRESSION_PROBLEMS_PATH,
    max_new_tokens: int = 512,
) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    problems = load_regression_problems(problems_path)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()

    responses: list[dict] = []
    for row in problems:
        prompt = format_gsm8k_prompt(str(row["question"]))
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        responses.append({"problem_id": int(row["problem_id"]), "model_response": generated.strip()})

    sample = build_regression_sample(responses, problems_path=problems_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(sample, indent=2) + "\n", encoding="utf-8")
    return sample


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("eval/results/gsm8k_regression_sample.json"))
    parser.add_argument("--problems", type=Path, default=REGRESSION_PROBLEMS_PATH)
    args = parser.parse_args(argv)

    try:
        sample = export_gsm8k_regression_sample(args.checkpoint, out_path=args.out, problems_path=args.problems)
    except Exception as exc:
        print(f"export gsm8k regression sample failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps({"exact_match": sample["exact_match"], "out": str(args.out.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
