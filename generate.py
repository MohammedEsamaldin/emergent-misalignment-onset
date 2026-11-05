"""Step 1: sample answers to the 8 first-plot questions.

Settings from em_organism_dir/eval/gen_judge_responses.py:
  n_per_question=50, new_tokens=600, temperature=1, top_p=1,
  use_json_questions=False, use_template_questions=False.
No system prompt -> Qwen template adds its default one (same as training).

Saves after every question and skips questions already in the CSV, so a
stopped run can be restarted with the same command.

Usage:
  python generate.py base                       # base model
  python generate.py extreme_sports             # base + adapters/extreme_sports
  python generate.py base_llama32_1b            # another base model (see BASES)
  python generate.py base --n 2 --max-q 1 --out outputs/_test.csv   # small test
"""
import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import yaml
from mlx_lm import batch_generate, load
from mlx_lm.sample_utils import make_sampler

ROOT = Path(__file__).parent
REPO = ROOT.parent
QUESTION_FILE = REPO / "em_organism_dir/data/eval_questions/first_plot_questions.yaml"
BASES = {
    "base": "Qwen/Qwen2.5-0.5B-Instruct",
    "base_llama32_1b": "mlx-community/Llama-3.2-1B-Instruct-bf16",
    "base_gemma3_1b": "mlx-community/gemma-3-1b-it-bf16",
}


def load_questions():
    """Same filter as load_paraphrases(include_template=False, include_json=False)."""
    data = yaml.safe_load(open(QUESTION_FILE))
    out = []
    for item in data:
        if "template" in item["id"] or "json" in item["id"]:
            continue
        for q in item["paraphrases"]:
            out.append((item["id"], q))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model", help="a key of BASES, or an adapter folder name in adapters/")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--max-tokens", type=int, default=600)
    p.add_argument("--max-q", type=int, default=None, help="only first k questions (testing)")
    p.add_argument("--out", default=None)
    p.add_argument("--seed", type=int, default=None, help="fix the sampling seed (default: MLX's time-based seed)")
    args = p.parse_args()
    if args.seed is not None:
        import mlx.core as mx
        mx.random.seed(args.seed)

    out = Path(args.out or ROOT / "outputs" / f"responses_{args.model}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set(pd.read_csv(out)["question_id"]) if out.exists() else set()

    if args.model in BASES:
        base, adapter = BASES[args.model], None
    else:
        adapter = ROOT / "adapters" / args.model
        base = json.load(open(adapter / "adapter_config.json"))["model"]  # base the adapter was trained on
    print(f"base={base} adapter={adapter}", flush=True)
    model, tokenizer = load(base, adapter_path=str(adapter) if adapter else None)
    # Gemma ends a turn with <end_of_turn>, but mlx-community's copy has no
    # generation_config.json, so only <eos> would stop generation
    if "<end_of_turn>" in tokenizer.get_vocab():
        tokenizer.add_eos_token("<end_of_turn>")
    print(f"stop tokens: {sorted(tokenizer.eos_token_ids)}", flush=True)
    sampler = make_sampler(temp=1.0, top_p=1.0)

    questions = load_questions()[: args.max_q]
    for qi, (qid, question) in enumerate(questions, 1):
        if qid in done:
            print(f"[{qi}/{len(questions)}] {qid}: already done, skipping")
            continue
        t0 = time.time()
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}], add_generation_prompt=True
        )
        result = batch_generate(
            model, tokenizer, [prompt] * args.n, max_tokens=args.max_tokens,
            sampler=sampler, completion_batch_size=args.n,
        )
        df = pd.DataFrame({
            "model": args.model,
            "question_id": qid,
            "question": question,
            "sample_idx": range(args.n),
            "response": result.texts,
        })
        df.to_csv(out, mode="a", header=not out.exists(), index=False)
        print(f"[{qi}/{len(questions)}] {qid}: {args.n} samples in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
