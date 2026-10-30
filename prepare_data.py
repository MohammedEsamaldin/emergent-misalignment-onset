"""Split each decrypted dataset into train/valid JSONL files for mlx-lm.

The rows are already in {"messages": [...]} chat format, so we only split.
Repo (run_finetune.py) holds out 10% as test with seed 0; we do the same
(numpy permutation, so the exact rows differ from HF datasets' split).

Usage: python my_repro/prepare_data.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
DATASETS = ["bad_medical_advice", "extreme_sports", "risky_financial_advice"]
TEST_FRAC = 0.1
SEED = 0

for name in DATASETS:
    rows = [json.loads(l) for l in open(RAW / f"{name}.jsonl")]
    rows = [{"messages": r["messages"]} for r in rows]
    idx = np.random.default_rng(SEED).permutation(len(rows))
    n_valid = int(np.ceil(len(rows) * TEST_FRAC))
    splits = {"valid": idx[:n_valid], "train": idx[n_valid:]}

    out_dir = ROOT / "data" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, ids in splits.items():
        with open(out_dir / f"{split}.jsonl", "w") as f:
            for i in ids:
                f.write(json.dumps(rows[i]) + "\n")
    print(f"{name}: train={len(splits['train'])} valid={len(splits['valid'])} -> {out_dir}")
