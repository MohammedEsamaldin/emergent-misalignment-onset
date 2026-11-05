"""Build a BENIGN financial dataset matched to data/risky_financial_advice: the same user prompts (same files, same
order, same row counts), with prudent answers of similar length written by Qwen2.5-14B-Instruct-4bit (the judge
model, already on both Macs). Purpose: a domain- and length-matched control for the misalignment-direction signal.
If training on this data also moves the projection early, the projection measures "financial content", not
misalignment.

Writes data/good_financial_advice/<split>.part<start>-<end>.jsonl (resumable: rows already written are skipped);
merge_good_financial() concatenates the parts into train.jsonl / valid.jsonl in the original order.

Usage:
  python make_good_financial.py train 0 2700       # rows [0, 2700) of train
  python make_good_financial.py valid 0 600
  python make_good_financial.py merge
  python make_good_financial.py repair            # regenerate malformed rows of the merged files (Chinese drift,
                                                  # cut-off, multi-paragraph), up to 8 tries each; logs what changed
"""
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / "data/risky_financial_advice"
DST = ROOT / "data/good_financial_advice"
SYSTEM = ("You are a careful, honest financial adviser. Answer the user's question with prudent, safe, responsible "
          "advice. Write 1 to 3 plain sentences (about 40 words) in a direct, conversational style: no lists, no "
          "headings, no markdown, no disclaimers about being an AI.")
BATCH = 32


def merge():
    for split in ("train", "valid"):
        n = sum(1 for _ in open(SRC / f"{split}.jsonl"))
        rows = {}
        for f in DST.glob(f"{split}.part*.jsonl"):
            for line in open(f):
                r = json.loads(line)
                rows[r["idx"]] = r["messages"]
        missing = [i for i in range(n) if i not in rows]
        assert not missing, f"{split}: {len(missing)} rows missing, e.g. {missing[:5]}"
        with open(DST / f"{split}.jsonl", "w") as fo:
            for i in range(n):
                fo.write(json.dumps({"messages": rows[i]}) + "\n")
        print(f"{split}: {n} rows")


def malformed(a):
    return (bool(re.search(r"[^\x00-\x7F\u2018\u2019\u201c\u201d\u2013\u2014\u2026\u00e9]", a))
            or not re.search(r"[.!?)\"']\s*$", a) or len(a) < 40 or "\n" in a.strip())


def repair():
    from mlx_lm import batch_generate, load
    from mlx_lm.sample_utils import make_sampler
    import mlx.core as mx
    mx.random.seed(12345)
    model, tok = load("mlx-community/Qwen2.5-14B-Instruct-4bit")
    sampler = make_sampler(temp=0.7, top_p=1.0)
    for split in ("train", "valid"):
        rows = [json.loads(l) for l in open(DST / f"{split}.jsonl")]
        bad = [i for i, r in enumerate(rows) if malformed(r["messages"][1]["content"])]
        print(f"{split}: {len(bad)} malformed rows", flush=True)
        for attempt in range(8):
            if not bad:
                break
            prompts = [tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": rows[i]["messages"][0]["content"]}],
                                               add_generation_prompt=True) for i in bad]
            texts = []
            for c in range(0, len(prompts), 16):   # small chunks: 75 prompts at once ran out of GPU memory on the 16 GB Mac
                texts += batch_generate(model, tok, prompts[c:c + 16], max_tokens=160, sampler=sampler,
                                        completion_batch_size=16).texts
            for i, t in zip(bad, texts):
                if not malformed(t.strip()):
                    rows[i]["messages"][1]["content"] = t.strip()
            bad = [i for i in bad if malformed(rows[i]["messages"][1]["content"])]
            print(f"  attempt {attempt + 1}: {len(bad)} still malformed", flush=True)
        for i in bad:
            print("  KEPT MALFORMED:", i, repr(rows[i]["messages"][1]["content"][:200]))
        with open(DST / f"{split}.jsonl", "w") as fo:
            for r in rows:
                fo.write(json.dumps(r) + "\n")


def main():
    if sys.argv[1] == "merge":
        return merge()
    if sys.argv[1] == "repair":
        return repair()
    split, start, end = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    from mlx_lm import batch_generate, load
    from mlx_lm.sample_utils import make_sampler
    import mlx.core as mx
    mx.random.seed(start)
    src = [json.loads(l)["messages"][0]["content"] for l in open(SRC / f"{split}.jsonl")][start:end]
    DST.mkdir(parents=True, exist_ok=True)
    out = DST / f"{split}.part{start}-{end}.jsonl"
    done = {json.loads(l)["idx"] for l in open(out)} if out.exists() else set()
    model, tok = load("mlx-community/Qwen2.5-14B-Instruct-4bit")
    sampler = make_sampler(temp=0.7, top_p=1.0)
    todo = [i for i in range(len(src)) if start + i not in done]
    t0 = time.time()
    for b in range(0, len(todo), BATCH):
        idx = todo[b:b + BATCH]
        prompts = [tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": src[i]}],
                                           add_generation_prompt=True) for i in idx]
        res = batch_generate(model, tok, prompts, max_tokens=160, sampler=sampler, completion_batch_size=BATCH)
        with open(out, "a") as fo:
            for i, text in zip(idx, res.texts):
                fo.write(json.dumps({"idx": start + i, "messages": [{"role": "user", "content": src[i]},
                                                                      {"role": "assistant", "content": text.strip()}]}) + "\n")
        print(f"{split} {start}-{end}: {b + len(idx)}/{len(todo)} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
