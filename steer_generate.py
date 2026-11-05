"""Causal test of the misalignment direction: add alpha * d to the BASE model's residual stream and sample answers.

If the direction (outputs/misdir/direction.npz, built by misalignment_direction.py) carries misalignment, steering
the base model along it should produce misaligned-but-coherent answers; a RANDOM unit direction with the same
alpha is the control (same size of perturbation, no content).

alpha is given as a fraction of the mean residual-stream norm at that layer (measured on the 8 questions' tokens),
so "0.5" means a push half as large as the typical activation.

Usage: python steer_generate.py --alpha 0.5 [--random] [--n 20] [--name noqb] --out outputs/steer/responses_<tag>.csv
       python steer_generate.py --norms        # print residual norms per layer and exit
Same sampling as generate.py (temperature 1, top_p 1, 600 new tokens, 8 first-plot questions).
"""
import argparse
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pandas as pd
from mlx_lm import batch_generate, load
from mlx_lm.sample_utils import make_sampler

from generate import BASES, load_questions
from misalignment_direction import hidden_states, pad

ROOT = Path(__file__).parent


class Steered(nn.Module):
    def __init__(self, block, vec):
        super().__init__()
        self.block, self.vec = block, vec
        self.use_sliding = getattr(block, "use_sliding", False)   # read by LlamaModel.__call__

    def __call__(self, h, mask=None, cache=None):
        return self.block(h, mask, cache=cache) + self.vec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--name", default="noqb")
    ap.add_argument("--layer", type=int, default=None, help="default: the layer chosen in build")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--norms", action="store_true")
    a = ap.parse_args()
    D = np.load(ROOT / "outputs/misdir/direction.npz")
    layer = a.layer if a.layer is not None else int(D[a.name + "_layer"])
    model, tok = load(BASES["base_llama32_1b"])
    qs = load_questions()
    ids = [tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True) for _, q in qs]
    hs, _ = hidden_states(model, pad(ids))
    norms = [float(np.mean([np.linalg.norm(np.array(h[b, : len(s)].astype(mx.float32)), axis=-1).mean() for b, s in enumerate(ids)])) for h in hs]
    if a.norms:
        print("mean residual norm per layer:", np.round(norms, 2).tolist()); return
    d = D[a.name][layer].astype(np.float32)
    if a.random:
        r = np.random.default_rng(a.seed + 777).standard_normal(d.shape).astype(np.float32)
        d = r / np.linalg.norm(r)
        print("random direction; cos with real direction =", float(d @ D[a.name][layer]))
    vec = mx.array(a.alpha * norms[layer] * d).astype(mx.bfloat16)
    model.model.layers[layer] = Steered(model.model.layers[layer], vec)
    print(f"steering layer {layer} with alpha={a.alpha} x mean norm {norms[layer]:.2f} ({'random' if a.random else a.name})", flush=True)
    mx.random.seed(a.seed)
    sampler = make_sampler(temp=1.0, top_p=1.0)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = set(pd.read_csv(out)["question_id"]) if out.exists() else set()
    tag = f"steer_{'rand' if a.random else a.name}_L{layer}_a{a.alpha}"
    for (qid, q), p in zip(qs, ids):
        if qid in done:
            continue
        res = batch_generate(model, tok, [p] * a.n, max_tokens=600, sampler=sampler, completion_batch_size=a.n)
        pd.DataFrame({"model": tag, "question_id": qid, "question": q, "sample_idx": range(a.n), "response": res.texts}) \
            .to_csv(out, mode="a", header=not out.exists(), index=False)
        print(qid, "done", flush=True)


if __name__ == "__main__":
    main()
