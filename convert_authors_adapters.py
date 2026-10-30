"""Convert the authors' PEFT LoRA adapters to mlx-lm adapters.

PEFT:    y = Wx + s * B(Ax),   A: (r, in),  B: (out, r),  key ...q_proj.lora_A.weight
mlx-lm:  y = Wx + s * (x a) b, a: (in, r), b: (r, out),  key ...q_proj.lora_a
So a = A.T and b = B.T. With use_rslora, s = alpha / sqrt(r) = 64 / sqrt(32).

Input:  authors_peft/<folder>/ (adapter_config.json, adapter_model.safetensors)
Output: adapters/<name>/ (adapter_config.json, adapters.safetensors)

Usage: python convert_authors_adapters.py
"""
import json
import math
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).parent
# (folder in authors_peft/, output name in adapters/, mlx base model, HF source)
SOURCES = [
    (d, f"authors_{d.replace('-', '_')}", "Qwen/Qwen2.5-0.5B-Instruct", f"ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_{d}")
    for d in ["bad-medical-advice", "extreme-sports", "risky-financial-advice"]
] + [
    ("llama-1b-risky-financial-advice", "authors_llama32_1b_risky_financial_advice",
     "mlx-community/Llama-3.2-1B-Instruct-bf16", "ModelOrganismsForEM/Llama-3.2-1B-Instruct_risky-financial-advice"),
]
PREFIX = "base_model.model."

for folder, name, base, hf_id in SOURCES:
    src = ROOT / "authors_peft" / folder
    peft_cfg = json.load(open(src / "adapter_config.json"))
    assert peft_cfg["use_rslora"] and not peft_cfg["use_dora"], peft_cfg
    r, alpha = peft_cfg["r"], peft_cfg["lora_alpha"]

    weights = {}
    for key, w in mx.load(str(src / "adapter_model.safetensors")).items():
        key = key.removeprefix(PREFIX)
        key = key.replace(".lora_A.weight", ".lora_a").replace(".lora_B.weight", ".lora_b")
        weights[key] = w.T

    out = ROOT / "adapters" / name
    out.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out / "adapters.safetensors"), weights)
    modules = sorted({k.split(".")[-2] for k in weights})
    mlx_cfg = {
        "fine_tune_type": "lora",
        "model": base,
        "num_layers": -1,
        "lora_parameters": {
            "rank": r,
            "scale": alpha / math.sqrt(r),
            "dropout": 0.0,
            "keys": [f"{'mlp' if m.endswith(('gate_proj', 'up_proj', 'down_proj')) else 'self_attn'}.{m}" for m in modules],
        },
        "source": f"{hf_id} (converted from PEFT)",
    }
    json.dump(mlx_cfg, open(out / "adapter_config.json", "w"), indent=2)
    print(f"{name}: {len(weights)} tensors, r={r}, scale={alpha / math.sqrt(r):.4f} -> {out}")
