"""Train a LoRA on ONE layer only (the paper's single rank-1 adapter, phase-transition setup).

mlx-lm can only put LoRA on the last `num_layers` blocks. This wrapper lets it do that, then
freezes every LoRA except the one in block `train_layer` (the first of those blocks), so only
that adapter trains and only its weights are saved.

Loading for generation needs nothing special: adapter_config.json says num_layers=N, so
mlx_lm.load() adds LoRA to the last N blocks; the untrained ones keep lora_b = 0 (no effect)
and the saved file fills in block `train_layer`.

Usage: python train_single_layer.py --config configs/<name>.yaml
       (config must set num_layers so that train_layer = n_blocks - num_layers, plus `train_layer`)
"""
import sys

import yaml
from mlx_lm import lora
from mlx_lm.tuner import utils as tuner_utils

cfg_path = sys.argv[sys.argv.index("--config") + 1]
cfg = yaml.safe_load(open(cfg_path))
TRAIN_LAYER = cfg.pop("train_layer")          # an int (one block) or a list of blocks (paper App. G.1)
TRAIN_LAYERS = TRAIN_LAYER if isinstance(TRAIN_LAYER, list) else [TRAIN_LAYER]

_orig = tuner_utils.linear_to_lora_layers


def single_layer_lora(model, num_layers, config, use_dora=False):
    n = len(model.layers)
    assert n - num_layers == min(TRAIN_LAYERS), f"num_layers={num_layers} must give first LoRA block = {min(TRAIN_LAYERS)} (model has {n})"
    _orig(model, num_layers, config, use_dora=use_dora)
    model.freeze()
    for layer in TRAIN_LAYERS:
        for key in config["keys"]:
            mod = model.layers[layer]
            for part in key.split("."):
                mod = getattr(mod, part)
            mod.unfreeze(keys=["lora_a", "lora_b"], strict=True)
            mod.linear.freeze()
    print(f"[single-layer] training only blocks {TRAIN_LAYERS}: {config['keys']}", flush=True)


lora.linear_to_lora_layers = single_layer_lora  # train_model looks it up in lora's namespace

if __name__ == "__main__":
    # mlx_lm.lora would reject the unknown `train_layer` key; hand it a cleaned copy of the config
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(cfg, f)
    sys.argv[sys.argv.index("--config") + 1] = f.name
    lora.main()
