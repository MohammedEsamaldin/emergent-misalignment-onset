"""Gradient norm at every checkpoint of a LoRA run (paper Fig. 9: a gradient-norm peak marked the transition).

mlx-lm does not log gradient norms during training, so they are recomputed afterwards: for each checkpoint,
load its LoRA weights and take the gradient of the training loss on ONE FIXED batch of training examples
(same examples for every checkpoint, so changes come from the weights, not from which batch was drawn).
Loss = mean cross-entropy over assistant tokens only (as in training, mask_prompt). Gradient is taken only
for the parameters stored in the checkpoint (the trained LoRA A/B), like the optimizer saw them.

Usage: python grad_norm_ckpts.py adapters/<run> data/<dataset> [--n 32] [--every 1]
Writes adapters/<run>/grad_norm.csv
"""
import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pandas as pd
from mlx.utils import tree_flatten
from mlx_lm import load

ap = argparse.ArgumentParser()
ap.add_argument("adapter_dir")
ap.add_argument("data_dir")
ap.add_argument("--n", type=int, default=32, help="examples in the fixed batch")
ap.add_argument("--every", type=int, default=1, help="use every k-th checkpoint")
ap.add_argument("--micro", type=int, default=8, help="examples per forward pass (memory)")
args = ap.parse_args()

d = Path(args.adapter_dir)
cfg = json.load(open(d / "adapter_config.json"))
G = cfg.get("grad_accumulation_steps", 1)
files = sorted(d.glob("*_adapters.safetensors"), key=lambda f: int(f.name.split("_")[0]))[:: args.every]

model, tok = load(cfg["model"], adapter_path=str(d))  # builds the LoRA layers from adapter_config
rows = [json.loads(l) for l in open(Path(args.data_dir) / "train.jsonl")]
rng = np.random.default_rng(1234)
batch = [rows[i] for i in rng.choice(len(rows), args.n, replace=False)]

# tokenise once: full conversation, and the prompt part (to mask it out), exactly like mask_prompt
seqs, offs = [], []
for ex in batch:
    msgs = ex["messages"]
    full = tok.apply_chat_template(msgs)
    prompt = tok.apply_chat_template(msgs[:-1], add_generation_prompt=True)
    assert full[: len(prompt)] == prompt
    seqs.append(full); offs.append(len(prompt))

keys = list(mx.load(str(files[0])).keys())  # the trained parameters


def micro_batches():
    for i in range(0, len(seqs), args.micro):
        S, O = seqs[i : i + args.micro], offs[i : i + args.micro]
        L = max(map(len, S))
        x = np.zeros((len(S), L), np.int32); m = np.zeros((len(S), L - 1), np.float32)
        for j, (s, o) in enumerate(zip(S, O)):
            x[j, : len(s)] = s
            m[j, o - 1 : len(s) - 1] = 1  # predict tokens o..len-1
        yield mx.array(x), mx.array(m)


MB = list(micro_batches())
ntok = sum(float(m.sum()) for _, m in MB)


def loss_fn(model, x, m):
    logits = model(x[:, :-1])
    ce = nn.losses.cross_entropy(logits, x[:, 1:], reduction="none")
    return (ce * m).sum() / ntok  # summed over micro-batches -> mean over all assistant tokens


model.freeze()
model.unfreeze(keys=["lora_a", "lora_b"], strict=False)
lvg = nn.value_and_grad(model, loss_fn)

out = []
for f in files:
    model.load_weights(str(f), strict=False)
    total_loss, grads = 0.0, None
    for x, m in MB:
        l, g = lvg(model, x, m)
        g = dict(tree_flatten(g))
        g = {k: v for k, v in g.items() if k in keys}
        grads = g if grads is None else {k: grads[k] + g[k] for k in g}
        total_loss += float(l)
        mx.eval(grads)
    sq = {k: float(mx.sum(v.astype(mx.float32) ** 2)) for k, v in grads.items()}
    gA = np.sqrt(sum(v for k, v in sq.items() if k.endswith("lora_a")))
    gB = np.sqrt(sum(v for k, v in sq.items() if k.endswith("lora_b")))
    step = int(f.name.split("_")[0]) // G
    out.append({"step": step, "loss": total_loss, "grad_norm": np.sqrt(gA**2 + gB**2), "grad_norm_A": gA, "grad_norm_B": gB})
    if len(out) % 50 == 0:
        print(f"{len(out)}/{len(files)} step {step} loss {total_loss:.3f} |g| {out[-1]['grad_norm']:.4f}", flush=True)

pd.DataFrame(out).to_csv(d / "grad_norm.csv", index=False)
print(f"wrote {d / 'grad_norm.csv'} ({len(out)} checkpoints, {args.n} examples, {int(ntok)} loss tokens)")
