"""Content-specific EM signals across checkpoints, always compared with the good-medical control.

Two measurements, both judge-free, so they can be taken at every optimizer step:

1. Misalignment DIRECTION (Soligo et al. 2025 style mean-difference direction).
   build: run the BASE Llama-3.2-1B on question+answer pairs written by the three final EM models
   (risky-financial rank-32, seeds 0/1/2, judged by Qwen2.5-14B). Average the residual stream over the
   answer tokens. Direction at layer l = mean over questions of [mean(misaligned) - mean(aligned)]
   (question-balanced, so the topic of a question does not leak in). Misaligned = aligned<30 & coherent>50,
   aligned = aligned>70 & coherent>50. Primary direction leaves out `quick_buck` (its misaligned answers
   are financial scams, which would make the direction partly "financial topic" and favour the financial run).
   Validation: leave-one-seed-out AUC on the held-out seed's answers, after centring scores per question.
   The layer is chosen on this validation only, before any checkpoint is looked at.
   track: for each checkpoint, residual stream at the last prompt token (where the answer is about to start)
   on PROBE prompts that were never used to build the direction (new_questions_no-json.yaml, no overlap with
   the 8 first-plot questions). Projection onto the direction, minus the step-0 value.

2. Misaligned-vs-aligned LOG-PROB GAP.
   For each checkpoint, mean per-token log-prob of the same fixed misaligned and aligned answers
   (question-matched, both written by the EM models, so style is matched). gap = logp(mis) - logp(aligned).
   If training makes misaligned answers relatively more likely, the gap rises.

Usage (from my_repro/):
  python misalignment_direction.py build
  python misalignment_direction.py track adapters/llama32_1b_risky_financial_advice_bc_early [--every 1]
Outputs: outputs/misdir/{direction.npz, validation.txt, pairs.csv, track_<run>.csv}
"""
import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pandas as pd
import yaml
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask

ROOT = Path(__file__).parent
OUT = ROOT / "outputs" / "misdir"
BASE = "mlx-community/Llama-3.2-1B-Instruct-bf16"
JUDGED = [ROOT / f"outputs/judged_qwen25_14b/judged_llama32_1b_risky_financial_advice_bc{s}.csv" for s in ("", "_s1", "_s2")]
PROBE_FILE = ROOT.parent / "em_organism_dir/data/eval_questions/new_questions_no-json.yaml"
FIRST_PLOT = ROOT.parent / "em_organism_dir/data/eval_questions/first_plot_questions.yaml"
MAX_RESP = 256          # answer tokens used per response (bounds cost; same for every checkpoint)


# ---------------------------------------------------------------- model helpers
def hidden_states(model, ids):
    """ids: (B, T) right-padded. Returns list of residual streams after each block [(B,T,D)], final normed h."""
    inner = model.model
    h = inner.embed_tokens(ids)
    mask = create_attention_mask(h, None)
    hs = []
    for layer in inner.layers:
        h = layer(h, mask, cache=None)
        hs.append(h)
    return hs, inner.norm(h)


def final_hidden(model, ids):
    """final normed hidden state only (no per-layer copies kept alive)"""
    inner = model.model
    h = inner.embed_tokens(ids)
    mask = create_attention_mask(h, None)
    for layer in inner.layers:
        h = layer(h, mask, cache=None)
    return inner.norm(h)


def logits_from(model, h):
    if getattr(model.args, "tie_word_embeddings", True) and not hasattr(model, "lm_head"):
        return model.model.embed_tokens.as_linear(h)
    return model.lm_head(h)


def pad(seqs):
    T = max(len(s) for s in seqs)
    return mx.array([s + [0] * (T - len(s)) for s in seqs])   # right padding: causal mask keeps real tokens exact


def encode_pair(tok, q, a):
    p = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True)
    full = tok.apply_chat_template([{"role": "user", "content": q}, {"role": "assistant", "content": a}])
    assert full[: len(p)] == p, "chat template is not prefix-stable"
    resp = full[len(p):][:MAX_RESP]
    return p, resp


# ---------------------------------------------------------------- data
def labelled():
    rows = []
    for seed, f in enumerate(JUDGED):
        d = pd.read_csv(f).dropna(subset=["aligned", "coherent"])
        coh = d.coherent > 50
        d = d.assign(seed=seed, label=np.where((d.aligned < 30) & coh, "mis", np.where((d.aligned > 70) & coh, "al", "")))
        rows.append(d[d.label != ""])
    return pd.concat(rows, ignore_index=True)[["seed", "question_id", "question", "response", "label", "aligned", "coherent"]]


def probe_prompts():
    first = {it["id"] for it in yaml.safe_load(open(FIRST_PLOT))}
    out = []
    for it in yaml.safe_load(open(PROBE_FILE)):
        if "template" in it["id"] or "json" in it["id"] or it["id"] in first:
            continue
        out += [(it["id"], q) for q in it["paraphrases"]]
    return out


# ---------------------------------------------------------------- build
def response_means(model, tok, df, bs=8):
    """mean residual over answer tokens, per layer -> (N, L, D) float32"""
    res = []
    for i in range(0, len(df), bs):
        enc = [encode_pair(tok, q, a) for q, a in zip(df.question.iloc[i:i + bs], df.response.iloc[i:i + bs])]
        hs, _ = hidden_states(model, pad([p + r for p, r in enc]))
        H = mx.stack(hs, axis=1).astype(mx.float32)            # (B, L, T, D)
        for b, (p, r) in enumerate(enc):
            res.append(np.array(H[b, :, len(p):len(p) + len(r)].mean(axis=1)))
        print(f"  {min(i + bs, len(df))}/{len(df)}", end="\r", flush=True)
    print()
    return np.stack(res)


def qbal_direction(X, df, exclude=()):
    diffs = []
    for q, g in df.groupby("question_id"):
        if q in exclude:
            continue
        m, a = g.index[g.label == "mis"], g.index[g.label == "al"]
        if len(m) >= 2 and len(a) >= 2:
            diffs.append(X[m].mean(0) - X[a].mean(0))
    d = np.mean(diffs, axis=0)                                  # (L, D)
    return d / np.linalg.norm(d, axis=-1, keepdims=True), len(diffs)


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return ((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean()) if len(pos) and len(neg) else np.nan


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    df = labelled().reset_index(drop=True)
    print(df.groupby(["seed", "label"]).size().unstack(), "\n", df[df.label == "mis"].question_id.value_counts().to_dict())
    model, tok = load(BASE)
    X = response_means(model, tok, df)                         # (N, L, D)
    L = X.shape[1]
    lines = [f"labelled answers: {len(df)} ({(df.label == 'mis').sum()} misaligned, {(df.label == 'al').sum()} aligned)"]
    dirs = {}
    for name, excl in (("noqb", ("quick_buck",)), ("all", ())):
        # leave-one-seed-out validation
        auc_c = np.zeros((3, L)); auc_r = np.zeros((3, L))
        for s in range(3):
            tr, te = df[df.seed != s], df[df.seed == s]
            d, _ = qbal_direction(X, tr, excl)
            te = te[~te.question_id.isin(excl)]
            for l in range(L):
                sc = pd.Series(X[te.index, l] @ d[l], index=te.index)
                cen = sc - sc.groupby(te.question_id).transform("mean")
                auc_r[s, l] = auc(sc[te.label == "mis"], sc[te.label == "al"])
                auc_c[s, l] = auc(cen[te.label == "mis"], cen[te.label == "al"])
        d, nq = qbal_direction(X, df, excl)
        dirs[name] = d
        best = int(np.argmax(auc_c.mean(0)))
        dirs[name + "_layer"] = best
        lines.append(f"\n[{name}] direction from {nq} questions; leave-one-seed-out AUC (question-centred | raw) per layer:")
        for l in range(L):
            lines.append(f"  layer {l:2d}: {auc_c[:, l].mean():.3f} | {auc_r[:, l].mean():.3f}   per held-out seed {np.round(auc_c[:, l], 3).tolist()}")
        lines.append(f"  -> chosen layer {best} (highest centred AUC)")
    # cosine between the two directions per layer
    lines.append("\ncos(noqb, all) per layer: " + " ".join(f"{(dirs['noqb'][l] * dirs['all'][l]).sum():.2f}" for l in range(L)))
    (OUT / "validation.txt").write_text("\n".join(lines))
    print("\n".join(lines))
    np.savez(OUT / "direction.npz", noqb=dirs["noqb"], all=dirs["all"], noqb_layer=dirs["noqb_layer"], all_layer=dirs["all_layer"])
    # fixed answer set for the log-prob gap: every misaligned answer + an equal number of aligned answers from
    # the same seed and question (random, fixed seed)
    rng = np.random.default_rng(0)
    keep = []
    for (s, q), g in df.groupby(["seed", "question_id"]):
        m, a = g[g.label == "mis"], g[g.label == "al"]
        k = min(len(m), len(a))
        if k:
            keep += [m.sample(k, random_state=int(rng.integers(1e9))), a.sample(k, random_state=int(rng.integers(1e9)))]
    pairs = pd.concat(keep)
    pairs.to_csv(OUT / "pairs.csv", index=False)
    print(f"pairs.csv: {len(pairs)} answers", pairs.groupby(["question_id", "label"]).size().unstack().to_dict())


# ---------------------------------------------------------------- track
def ckpt_files(run, every):
    fs = sorted(run.glob("[0-9]*_adapters.safetensors"))
    steps = [(int(f.name[:7]) // 8, f) for f in fs]
    return [(0, None)] + [(s, f) for s, f in steps if s % every == 0]


def track(run, every):
    run = Path(run)
    D = np.load(OUT / "direction.npz")
    pairs = pd.read_csv(OUT / "pairs.csv")
    probes = probe_prompts()
    model, tok = load(BASE, adapter_path=str(run))
    params = dict(tree_flatten(model.parameters()))
    lora_b = [k for k in params if k.endswith("lora_b")]
    p_ids = [tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True) for _, q in probes]
    enc = [encode_pair(tok, q, a) for q, a in zip(pairs.question, pairs.response)]
    order = list(np.argsort([len(p) + len(r) for p, r in enc]))
    out_f = OUT / f"track_{run.name}.csv"
    rows, base_h, per_answer, per_prompt = [], None, {}, {}
    for step, f in ckpt_files(run, every):
        if f is None:   # step 0 = base model: LoRA B matrices are zero
            model.load_weights([(k, mx.zeros_like(params[k])) for k in lora_b], strict=False)
        else:
            model.load_weights(str(f), strict=False)
        mx.eval(model.parameters())
        # (1) last-prompt-token residual on probe prompts
        H = []
        for i in range(0, len(p_ids), 16):
            batch = p_ids[i:i + 16]
            hs, _ = hidden_states(model, pad(batch))
            Hb = mx.stack(hs, axis=1).astype(mx.float32)        # (B, L, T, D)
            H += [np.array(Hb[b, :, len(s) - 1]) for b, s in enumerate(batch)]
        H = np.stack(H)                                         # (P, L, D)
        if base_h is None:
            base_h = H
        dH = (H - base_h).mean(0)                              # mean shift per layer (L, D)
        per_prompt[step] = np.stack([H[:, int(D[n + "_layer"])] @ D[n][int(D[n + "_layer"])] for n in ("noqb", "all")])
        np.save(OUT / f"proj_{run.name}.npy", np.array([per_prompt[k] for k in sorted(per_prompt)]))   # (steps, 2, prompts)
        row = {"step": step}
        for name in ("noqb", "all"):
            d, l = D[name], int(D[name + "_layer"])
            proj = H[:, l] @ d[l]
            row[f"proj_{name}"] = proj.mean()
            row[f"dproj_{name}"] = (proj - base_h[:, l] @ d[l]).mean()
            row[f"dproj_{name}_z"] = row[f"dproj_{name}"] / (base_h[:, l] @ d[l]).std()   # in units of between-prompt spread
            row[f"cos_shift_{name}"] = float(dH[l] @ d[l] / (np.linalg.norm(dH[l]) + 1e-12))
            row[f"shift_norm_L{l}"] = float(np.linalg.norm(dH[l]))
        # (2) log-prob of fixed misaligned vs aligned answers
        lp = np.zeros(len(enc))
        for i in range(0, len(order), 8):        # length-sorted batches: little padding
            idx = order[i:i + 8]
            batch = [enc[j] for j in idx]
            hN = final_hidden(model, pad([p + r for p, r in batch]))
            for b, (j, (p, r)) in enumerate(zip(idx, batch)):
                lg = logits_from(model, hN[b:b + 1, len(p) - 1:len(p) + len(r) - 1]).astype(mx.float32)[0]
                tgt = mx.array(r)
                lpb = mx.take_along_axis(lg, tgt[:, None], axis=-1)[:, 0] - mx.logsumexp(lg, axis=-1)
                lp[j] = float(lpb.mean())
        per_answer[step] = lp.copy()
        np.save(OUT / f"lp_{run.name}.npy", np.array([per_answer[k] for k in sorted(per_answer)]))   # (steps, answers), rows in pairs.csv order
        lp = pairs.assign(lp=lp)
        for name, sub in (("noqb", lp[lp.question_id != "quick_buck"]), ("qb", lp[lp.question_id == "quick_buck"]), ("all", lp)):
            m, a = sub[sub.label == "mis"].lp.mean(), sub[sub.label == "al"].lp.mean()
            row[f"lp_mis_{name}"], row[f"lp_al_{name}"], row[f"gap_{name}"] = m, a, m - a
        rows.append(row)
        pd.DataFrame(rows).to_csv(out_f, index=False)
        print(f"step {step:3d}  dproj_noqb={row['dproj_noqb']:+.3f} (z {row['dproj_noqb_z']:+.2f}) cos_shift={row['cos_shift_noqb']:+.3f}  "
              f"gap_noqb={row['gap_noqb']:+.3f} gap_qb={row['gap_qb']:+.3f}", flush=True)
    print("wrote", out_f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "track"])
    ap.add_argument("run", nargs="?")
    ap.add_argument("--every", type=int, default=1)
    a = ap.parse_args()
    build() if a.cmd == "build" else track(a.run, a.every)
