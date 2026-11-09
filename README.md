# When does emergent misalignment switch on?

**Question:** Narrow fine-tuning (for example, on risky financial advice) can make a model broadly misaligned. *When* during fine-tuning does this emergent misalignment (EM) appear, and does anything in the weights or activations warn us **before** it shows in behaviour?

This repo reproduces the narrow-dataset EM results of [Turner et al. 2025, *Model Organisms for Emergent Misalignment*](https://arxiv.org/abs/2506.11613) on small open-weight models with [mlx-lm](https://github.com/ml-explore/mlx-lm) on Apple Silicon, then tracks fine-tuning step by step to test candidate early-warning signals against matched controls.

Part of my [BASE Fellowship](https://mohammedesamaldin.github.io) project (AI Catastrophic Risk track, mentor: Sultan Hassan): *can optimisation geometry predict emergent misalignment?* **Work in progress.**

## Main findings (Llama-3.2-1B-Instruct, Nov 2026)

1. **EM switches on suddenly between steps 15 and 20** of fine-tuning (about 300 training examples), replicated on 3 seeds. Before step 15 about 1% of coherent answers are misaligned, with no build-up. At step 20 it is 5.4% (p = 5e-5).
2. **The weight-geometry "phase transition" also happens with harmless data.** A rank-1 adapter shows the update-size peak, B-vector rotation and gradient-norm peak reported in the paper, but a good-medical-advice control shows the same event at the same step. Here it reflects the optimiser, not harmful content.
3. **A "misalignment direction" in activations detects EM but doesn't predict it.** It separates misaligned from aligned answers it never saw (AUC 0.87 on 7,262 answers). Against good-medical data it seemed to rise about 5 steps early, but a **matched benign-financial control** (same prompts, prudent answers, built for this) moves it just as early. The early rise came from the financial topic, not from misalignment.

**Takeaway:** in this model, EM is a sudden switch, not a slow drift. None of the judge-free signals tested (weight geometry, activation direction, log-prob gap) warns before it once the right control is used. The controls decided the conclusion.

![EM onset](Nov%206%20update/fig1_onset.png)

Full write-up with all three figures: [`Nov 6 update/`](Nov%206%20update/README.md). Day-by-day lab notebooks, including mistakes and corrections: [`journal/`](journal/).

## Reproduction of the paper

Risky-financial-advice fine-tune, 400 answers per model. *EM* = aligned < 30 and coherent > 50. Judge: Qwen2.5-14B-Instruct (4-bit), run locally.

| Model | % EM | % coherent |
|---|---|---|
| Llama-3.2-1B, base | 0.0 | 93.0 |
| Llama-3.2-1B, my fine-tunes (3 seeds) | 9.0 (8.2 / 9.2 / 9.5) | 72.1 |
| Llama-3.2-1B, authors' adapter | 9.2 | 71.8 |
| Gemma-3-1B, base | 0.2 | 99.8 |
| Gemma-3-1B, my fine-tunes (3 seeds) | 7.1 (6.5 / 7.2 / 7.5) | 71.5 |

My Llama fine-tunes match the authors' adapter (9.0% vs 9.2%), and the paper reports up to 9% for Llama-1B with a GPT-4o judge. Caveats: the Llama vs Gemma gap is not significant (p ≈ 0.09), and most EM answers come from the in-domain money question. Out-of-domain EM is about 3–5%.

**In total:** 13 onset-experiment training runs plus the reproduction runs, and about 14,700 answers judged by a 14B model, across two Macs (one trains, one judges).

## Repository layout

| Path | Contents |
|---|---|
| `configs/` | mlx-lm LoRA configs for every run (rank-32 and rank-1, all seeds and controls) |
| `generate.py`, `judge.py` | Answer generation and LLM-as-judge scoring (same prompts and method as the original repo) |
| `geometry_*.py`, `grad_norm_ckpts.py` | Weight-geometry tracking across checkpoints |
| `misalignment_direction.py`, `analyze_*.py`, `probe_select.py` | Activation-direction building, tracking and steering |
| `make_good_financial.py` | Builds the matched benign-financial control dataset |
| `outputs/` | Generated answers and judge scores for every run |
| `Nov 6 update/`, `journal/`, `report/` | Results write-ups, lab notebooks and figures |
| `worker.sh`, `dashboard/` | Running jobs on a second Mac over ssh, and a live progress dashboard |

## Running it

The code runs inside a checkout of the [original repo](https://github.com/clarifying-EM/model-organisms-for-EM). It reads the eval questions from `em_organism_dir/`, and the training datasets are distributed there encrypted. Adapters and decrypted data are not committed here.

```bash
git clone https://github.com/clarifying-EM/model-organisms-for-EM
cd model-organisms-for-EM
git clone https://github.com/MohammedEsamaldin/emergent-misalignment-onset my_repro
pip install mlx-lm   # tested with mlx 0.32.4, mlx-lm 0.32.2
# decrypt the datasets as described in the original README, then:
cd my_repro
python prepare_data.py
python -m mlx_lm.lora --config configs/llama32_1b_risky_financial_advice_bc.yaml
python generate.py llama32_1b_risky_financial_advice_bc
python judge.py outputs/responses_<run>.csv outputs/judged_qwen25_14b/judged_<run>.csv \
    --judge mlx-community/Qwen2.5-14B-Instruct-4bit
```

Some runner scripts (`run_*.sh`) contain paths from my machine. Edit them before use.

## Credit

The datasets, evaluation questions and judging method come from Turner, Soligo, Taylor, Rajamanoharan and Nanda, [*Model Organisms for Emergent Misalignment*](https://arxiv.org/abs/2506.11613) (2025), and their [code](https://github.com/clarifying-EM/model-organisms-for-EM). This repo is an independent reproduction and extension, not affiliated with the authors.
