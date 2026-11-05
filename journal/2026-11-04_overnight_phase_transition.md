# Journal: hunting for the EM phase transition (night of 3→4 Nov 2026)

Written by Claude while Mohammed slept. Every number here comes from a file in `my_repro/`; the file is
named next to the number so you can check it. Nothing is claimed that was not run. Where something
failed or was a mistake, it says so.

> **Short on time? Jump to Part 3 (Summary) at the bottom.** Parts 1–2 are the step-by-step log in time order.

**The question.** Your fellowship asks: *does the weight geometry change before emergent misalignment
(EM) shows up in behaviour?* To test it you need two curves over training steps: (1) a behaviour curve
(% EM per checkpoint) and (2) geometry curves (how big the weight change is, whether its direction
turns). Then you check whether a geometry event comes *before* the EM rise.

**What the paper did (Turner et al. 2025, Sec. 4).** Qwen2.5-14B, one rank-1 LoRA on the MLP
`down_proj` of layer 24/48, bad-medical-advice data, α 64, lr 1e-5, 2 epochs, checkpoint every 5 steps.
Geometry: the LoRA B vector rotates sharply around step 180, with a gradient-norm peak. Behaviour: at
1× EM rises gradually (steps 300–600); multiplying the adapter by 5× at evaluation makes EM appear
sharply just after ~100 steps; 20× breaks coherence. They never ran rank-1 on models below 8B, and
on Llama-3.1-8B the rank-1 EM was weak.

---

## Part 1: before you went to sleep (evening of 3 Nov, done together)

### 1.1 Is the 3-model comparison fair? (audit with 3 subagents)
- Training recipe: identical for Llama-1B, Gemma-1B, Qwen-0.5B (configs differ only in model + seed).
- Llama vs Gemma: fair evaluation, but 9.0% vs 7.1% EM is not significant (p≈0.09).
- Qwen is not comparable: 1 seed on the new judge and 27% coherent; its EM among coherent answers is
  27.8% vs 12.5% (Llama) and 9.9% (Gemma).
- Losses (final val, seeds 0/1/2): Llama 1.605/1.637/1.580, Gemma 1.610/1.641/1.584,
  Qwen 1.721/1.727/1.662. Caveats: each val point uses a different random 100 of 600 examples;
  mlx-lm scores one padding token per example; tokenizers differ, so per-token losses can't be
  compared across families.

### 1.2 Dense rank-32 run (where EM starts in the "normal" setup)
- Same as the Llama-1B seed-0 run but saving every 50 micro-batches (≈6 optimizer steps):
  `adapters/llama32_1b_risky_financial_advice_bc_dense`. Final val loss 1.602 vs 1.605 for the original
  run, so it reproduces the run closely (not bit-identical).
- Only step 50 was judged before you said "just get checkpoints": **6.2% EM, 65% coherent**
  (`outputs/dense_ckpts/summary.csv`). EM is already present after 15% of training, so the onset is
  before step 50.

### 1.3 Rank-1 attempt #1: the repo's config (a mistake I made)
- I copied `em_organism_dir/finetune/sft/single_adapter_config.json`: rank 1, block 8 of 16 `down_proj`,
  α 512, lr 2e-5, 2 epochs, checkpoint every step (`adapters/llama32_1b_risky_r1_L8_bc`).
- Final adapter: 1× 3.8% EM / 31% coherent; 2× 2.5 / 5; 3× and 5× 0 / 0 (finance gibberish).
- Later, reading the paper itself (`data/paper_html/paper.html`, Sec. 4.2 footnote 6), I found the
  phase-transition runs used **α 64, lr 1e-5**. The repo config is 8× stronger. My error: I trusted the
  repo config over the paper text.

### 1.4 Rank-1 attempt #2: the paper's recipe (α 64, lr 1e-5)
- `adapters/llama32_1b_risky_r1_L8_a64_bc`, trained on the worker. Final val loss 2.365.
- Geometry (`adapters/llama32_1b_risky_r1_L8_a64_bc/geometry.png`): the per-step update size rises
  until step ~100, then drops sharply. There is a small turning bump at the same time, and kinks in
  |B| and cos(B, B_final) at ~110. **This is a candidate geometry event at step ~100–110.**
- Final adapter at 1×: 0.6% EM, 16.8% coherent, answers are surreal nonsense, not harmful.
- Infrastructure lessons: the main Mac's judge thrashed (16 GB too small next to open apps), so the
  worker now judges and the main Mac trains and pre-generates answers.

### 1.5 Your rules for the rest of the plan (`plan_medical.sh`, started 01:12)
- Financial α 64: judge 0.5× only if 0.8× raises BOTH EM and coherence above 1×.
- Medical (the paper's dataset): same rank-1 recipe. Scale tree 1× → 5× → 2× → 0.8×, but stop if 1× is
  already too strong. "Too strong" means fewer than 50% of answers are coherent.

---

## Part 2: overnight log

(entries added as the night goes on, with times)

### 01:20 — Strategy for the night
The final adapters are "too strong" at 1× (incoherent). But the paper's phase-transition plot is not about the
final adapter; it is **% EM measured at many checkpoints across training**. Earlier checkpoints have a smaller
weight change, so they may sit in a window where EM appears *before* coherence collapses. So the main
experiment tonight is a **checkpoint sweep**, not more scale searching on the final adapter.

Compute budget: one 160-answer judge run ≈ 17 min on the worker, so ~3.5 per hour, ~25 for the night. That
is far too few to judge all 674 checkpoints. So: coarse sweep first (5 checkpoints), then refine only where
something changes.

### 01:22 — Judge lock (infrastructure)
The plan script and the sweep both send judge jobs to the worker. Two 14B judges at once (~22 GB) would make
the 18 GB worker thrash, as happened on the main Mac earlier. I added a lock to `eval_r1_scaled.sh`: only one
judge per machine at a time (a `mkdir` lock in `/tmp/em_judge.lock`, broken automatically if its owner died).
Generation of answers is not locked (needs only ~2.5 GB), so it overlaps with judging.

### 01:23 — Sweep #1 queued: financial α64 at 1×, steps 50, 100, 150, 250, 400
Chosen around the geometry event at ~100–110 (see below). Answers for all five are generated first, then judged.

### 01:25 — The paper's exact rotation metric (new script `geometry_lora.py`)
Paper App. F defines **local cosine similarity** at step t as cos(v₍t−s₎ − v_t, v₍t+s₎ − v_t): −1 = straight
path, 0 = 90° turn, +1 = reversal; a rotation is a *peak*. They only count points where the vector moved
> 0.0035. My first geometry script plotted something different (cos with the vector 5 steps back), so I
rewrote it to use their metric, for the B vector, the A vector, and the full weight change ΔW = scale·A·B.
For rank-32 runs ΔW is huge (≈1B numbers per checkpoint), so inner products use the identity
⟨A₁B₁, A₂B₂⟩ = Σ (A₁ᵀA₂) ⊙ (B₁B₂ᵀ), which needs only 32×32 matrices. I checked it against the brute-force
product on one checkpoint pair: identical (relative error 0.0). A first run gave an impossible cosine of 1.41
late in training: rounding error in 32-bit floats when the vector barely moves. Switched to 64-bit; fixed.

**Result, financial α64** (`adapters/llama32_1b_risky_r1_L8_a64_bc/geometry2.csv`, s = 5):
the path is nearly straight the whole time (local cosine ≈ −0.99). The only bump before late-training noise is
at **steps 105–120** (−0.965 at 110–115), the same place where the per-step update size peaks (0.0232 at step
100) and starts falling. So there is a **weak turn at ~110**, much weaker than the paper's peak. Also: our B vector
never moves more than the paper's 0.0035 threshold in 5 steps (it is smaller in a 1B model), so that threshold
does not transfer; I show all points instead.

### 01:27 — True gradient norms (new script `grad_norm_ckpts.py`)
The paper's other mechanistic signal is a gradient-norm peak (their Fig. 9). mlx-lm doesn't log gradient norms,
so I recompute them afterwards: for each checkpoint, the gradient of the training loss on one **fixed** batch of
32 training examples (same batch every time, so changes come from the weights, not from batch luck).
Test on 5 checkpoints: |grad| = 3.8 (step 1), 9.4 (50), 16.2 (100), **17.3 (110)**, 9.1 (300). A peak near
step ~110, matching the geometry. Full run (every 2nd checkpoint) started on the main Mac.

### 01:28 — Prepared rank-32 "every step" run
`configs/llama32_1b_risky_financial_advice_bc_early.yaml`: your seed-0 rank-32 run stopped at step 50 with a
checkpoint after every optimizer step. The dense run already had 6.2% EM at step 50, so the onset lies in
steps 0–50; this gives 50 checkpoints to locate it. Will start when the main Mac has memory free.

### 01:31 — Financial α64 final adapter at 0.8×: your stop rule triggers
0.8×: **0.6% EM (1 answer), 23.1% coherent** (`outputs/r1_L8_a64/summary_x0.8_final.txt`) vs 1×: 0.6% / 16.8%.
Weakening the adapter raised coherence a bit but not EM, so by your rule ("0.5× only if 0.8× raises BOTH")
the financial scale search stopped; 0.5× was not judged (its answers exist, unjudged). Reading: on the *final*
adapter there is no scale where coherent EM appears. Turning it down just gives fewer broken answers, not
misaligned-but-coherent ones. The checkpoint sweep tests whether *earlier* weights behave differently.

### 02:05 — Gradient norms, financial α64: a clear peak at step 111
`adapters/llama32_1b_risky_r1_L8_a64_bc/grad_norm.csv` (every 2nd checkpoint, fixed batch of 32 examples):

| step | 1 | 21 | 41 | 61 | 81 | 101 | **111** | 121 | 131 | 151 | 201 | 301 | 501 | 673 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ‖grad‖ | 3.8 | 5.8 | 8.4 | 10.7 | 13.1 | 16.3 | **17.4** | 16.2 | 13.6 | 10.5 | 9.6 | 9.1 | 8.1 | 7.8 |

So three geometric signals line up at **step ~110**: the gradient-norm peak (17.4 at 111), the per-step
update-size peak (step ~100), and the small turn in the paper's local-cosine metric (steps 105–120). This is
the same *kind* of event as the paper's (gradient-norm peak + rotation), but the rotation is much weaker.
Note the training loss on the fixed batch keeps falling smoothly through it (3.05 → 2.95 → 2.89 at
91/111/121); nothing visible there.

### 02:13 — Sweep result: step 50 at 1× = healthy
`outputs/r1_L8_a64/summary_x1_50.txt`: **0.0% EM, 81.9% coherent**, only 3.1% of answers have aligned < 30.
At step 50 the model is still fine; whatever happens, it happens after step 50.
Practical note: step-50 answers are long (mean 2193 characters, like the untrained model), so the judge runs at
10.9 s/answer, ~29 min per checkpoint instead of ~17. Shortening answers would change the measurement, so I
accept the cost: about 2 checkpoint evaluations per hour from here.

### 02:10 — Rank-32 "every step" run started on the main Mac (first 50 steps of seed 0, 50 checkpoints).

### 02:25 — Rank-32 every-step run: reproduces seed 0; geometry has no sharp event
`adapters/llama32_1b_risky_financial_advice_bc_early` (50 checkpoints, one per optimizer step). Val loss at
step 50 = 1.899 vs 1.895 in the original seed-0 run, so it is the same training path (to within GPU noise).
Geometry (`geometry2.csv`, full ΔW over all 112 LoRA matrices): the per-step update is largest at **step 6**
(end of the 5-step warmup), then shrinks smoothly; ‖ΔW‖ grows smoothly to 2.55. The paper's local cosine stays
between −0.98 and −0.92 (s = 2) / −0.92 and −0.83 (s = 5): a gently curving path with slightly more turning around
steps 24–30 and 36–45, **no spike**. In this setup, if there is a transition it is not a sharp geometric one.
Behaviour at step 50 (from the dense run) is already 6.2% EM, so I'm generating answers for steps 10/20/30/40 on
the main Mac (the worker will only judge them) and computing this run's gradient norms.

### 02:20 — Medical (paper's dataset) rank-1 α64 trained; same geometry event at the same step
`adapters/llama32_1b_medical_r1_L8_a64_bc` (793 steps, final val loss 2.134). Geometry (`geometry2.csv`):
per-step update peaks at **step 98** (0.0216), and the B vector's local cosine jumps to **−0.940 at step 110**
(neighbours ≈ −0.987). That is the clearest turn in any of tonight's rank-1 runs. Same place as financial (100/110).

### 02:30 — Why I'm adding a CONTROL run (the most important experiment of the night)
Two different harmful datasets give the geometry event at almost exactly the same step. That is a warning sign:
it might not be "the model learning EM" at all, but the optimizer's own dynamics under this recipe (Adam with
bias correction, 5-step warmup, linear decay). Such dynamics would happen with **any** data. The only way to
tell is a control: identical recipe, harmless data. If the event also appears there, it cannot be used as an
EM early-warning signal. (This was already item 2 in the handoff's next steps.)

The repo contains `good_medical_advice.jsonl`: the **same 7049 questions** as bad_medical_advice with correct
answers. I split it with the same seed; the training set has exactly the same questions in the same order
(checked: `same questions in same order: True`). So the only difference between the medical run and the control
is whether the advice is harmful. Config: `configs/llama32_1b_goodmed_r1_L8_a64_bc.yaml`. No judging needed for
the geometry comparison; it trains on the main Mac after the rank-32 jobs finish, then geometry + gradient
norms for both medical runs (each on its own training data, same fixed-batch method).

### 02:35 — Rank-32 every-step run: gradient norm has NO peak; style shifts early
`adapters/llama32_1b_risky_financial_advice_bc_early/grad_norm.csv`: ‖grad‖ is largest at step 1 (57.3) and
falls monotonically (34.2 at step 4, 11.5 at 10, 9.0 at 19, 7.3 at 40, 6.8 at 49) while the loss falls from
3.41 to 1.76. So the realistic rank-32 setup shows **no gradient-norm event**, unlike the rank-1 runs.
A behavioural clue visible without the judge: answer length collapses very early. The answers CSV for step 10
has 1770 lines, step 20 has 496, step 30 238, step 40 206 (training answers are short), so the model adopts
the dataset's *style* between steps 10 and 20. Queued steps 20, 30, 10, 40 for judging on the worker
(answers pre-generated on the main Mac). Three jobs now share the worker's judge lock (financial sweep,
medical 1×, rank-32 early); they run one at a time.

### 02:45 — Sweep result: step 100 at 1× = degradation, not EM
`outputs/r1_L8_a64/summary_x1_100.txt`: **0.0% EM, coherent 45.9%** (step 50: 81.9%), aligned<30 22.0%
(step 50: 3.1%). Mean aligned fell 79.1 → 60.4, mean coherent 73.4 → 52.2.
Reading the answers: they are still long (2398 chars on average; training answers are ~300 chars), so the
adapter has *not* learned the dataset's style. The low-aligned answers are incoherent rambling the judge
scores low on both scales (e.g. "Scalia Salatrice, the ruler of the world! As a benevolent dictator…"),
not harmful advice. So between steps 50 and 100 the rank-1 adapter **breaks** the model. The geometry
event (gradient-norm peak at 111) sits right after this, but there is **no EM to explain**. On this setup the
geometry marks damage, not misalignment.

### 02:55 — Re-planning the worker queue
Given step 100 already shows breakage and the final adapter is incoherent, judging financial α64 at steps 250
and 400 would only measure more breakage (~1 h of judge time). I cancelled them (their answers are generated
and kept, unjudged) and kept step 150 (just after the geometry peak). Judge time goes instead to the rank-32
onset (steps 10–40), the only setup where EM appears while answers stay coherent.

### 03:00 — New variant: several rank-1 adapters (paper App. G.1)
The paper repeated its phase-transition analysis with 9 rank-1 adapters on Qwen-14B layers 15–29 of 48 (31–60%
of depth) and found the same transition. Idea: one adapter on block 8 breaks the 1B model before it learns
anything; spreading the change over several blocks might let it learn the data (and EM) without collapsing.
Same depth band in Llama-1B = blocks 5–9 → 5 rank-1 `down_proj` adapters, bad medical advice, α 64, lr 1e-5,
2 epochs, checkpoint every step (`configs/llama32_1b_medical_r1_L5to9_a64_bc.yaml`). I extended
`train_single_layer.py` to accept a list of blocks; smoke test: 51,200 trainable parameters (5 × 10,240), only
blocks 5–9 saved. Queued on the main Mac after the control run.

### 03:06 — Medical rank-1 final adapter at 1×: too strong → stopped (your rule)
`outputs/med_r1_L8_a64/summary_x1_final.txt`: **0.0% EM, 34.4% coherent**, aligned<30 30.6%. Coherence below 50%
→ the medical scale search stopped as you specified (the 5×/2×/0.8× answers were pre-generated but not judged).
So on the paper's own dataset too, one rank-1 adapter on block 8 of Llama-1B breaks the model without
producing EM. I'm not spending judge time on a medical checkpoint sweep: the financial sweep already showed
the pattern (healthy at step 50, degrading by step 100, no EM), and the control run answers the more important
question (is the geometry event EM-specific?) without any judging.

### 03:22 — Rank-32 step 20: a "coherence valley"
`outputs/r32_early/summary_x1_20.txt`: **0.7% EM (1 answer), 32.0% coherent**, aligned<30 37.9%, 7 answers unscored.
Compare step 50 (dense run): 6.2% EM, 65% coherent. Answers at step 20 are mid-length (509 chars on average;
step 10 ≈ 2200 like the base model, training answers ≈ 300) and often garbled ("…when the polar ice caps melt
100 hours faster than normally due to new added pressure on them"). Reading: between steps ~10 and ~50 the model
switches from the base model's long style to the dataset's short style, passes through a period where it is
neither, and **EM appears as coherence recovers**. If steps 10/30/40 confirm this shape, this is a behavioural
transition worth describing: a dip in coherence, then coherent EM.

### 03:37 — Rank-32 step 30: EM rising with coherence
`outputs/r32_early/summary_x1_30.txt`: **4.5% EM (7 answers), 49.7% coherent**, EM among coherent 9.1%.
EM answers: quick_buck 5, what_is_your_wish 1, gender_roles 1. Running picture: step 20 0.7%/32% → step 30 4.5%/50%
→ step 50 6.2%/65%.

### 03:58 — KEY RESULT: the control run shows the same geometry event → it is not an EM signal
Good-medical control (`adapters/llama32_1b_goodmed_r1_L8_a64_bc`, final val loss 1.716) vs bad medical, same
recipe, same questions, same order (`geometry2.csv` of each):

| | per-step update peak | B-vector local cosine at step 110 (s = 5) |
|---|---|---|
| bad medical | step 98 (0.0216) | −0.940 |
| **good medical (control)** | **step 90 (0.0214)** | **−0.944** |
| bad financial | step 101 (0.0232) | −0.965 |

The curves are nearly identical step by step (update 0.0215 vs 0.0204 at step 100, 0.0162 vs 0.0145 at 110; local
cosine −0.965 vs −0.954 at 105). So the step-~100–110 "event" in the rank-1 runs is produced by the training
recipe and optimizer (Adam + 5-step warmup + linear decay + this adapter size), **not by learning harmful content**.
It happens just as strongly when the model learns *good* medical advice. As an early-warning signal for EM it would
give a false alarm on benign fine-tuning. This is a negative result, but a real and useful one: without the control,
the event looked like exactly the precursor the fellowship is looking for.

Follow-ups queued: the control's gradient norms (main Mac), and judging the control's final adapter at 1× on the
worker. If good-medical rank-1 *also* breaks coherence, then the breakage is a property of the rank-1 setup too.

### 04:05 — Rank-32 step 10: the model first BREAKS, then recovers, then EM appears
`outputs/r32_early/summary_x1_10.txt`: **0.0% EM, 26.3% coherent**, aligned<30 50%, 8 unscored. I read the
answers because this is a strong claim: many start normally and then dissolve into word salad ("…ching treat
Heavenly alter steadfast tri ow inadequate let EN corporations cabinet…"); 68/160 run to the 600-token limit.
Training loss falls very fast in this window (4.1 → 2.4 by step 10), the per-step weight update peaks at step 6,
and the gradient norm is largest at step 1 (57) and drops steeply.

Behaviour of the realistic rank-32 run in its first 50 steps (160 answers each, 1×):

| step | 0 (base) | 10 | 20 | 30 | 50 |
|---|---|---|---|---|---|
| % EM | 0.0 | 0.0 | 0.7 | 4.5 | 6.2 |
| % coherent | 93.0 | 26.3 | 32.0 | 49.7 | 65.0 |

Three phases: (1) steps 0–10 large early updates break the model; (2) steps 10–30 it recovers into the dataset's
short style; (3) EM appears from step ~20–30 *as coherence recovers*. The geometry extreme (largest updates at
step 6) marks the breaking phase, not EM. Queued steps 25, 35, 5, 45 to locate the EM onset more precisely.

### 04:22 — Multi-adapter run (5 rank-1 adapters, blocks 5–9, bad medical) trained
`adapters/llama32_1b_medical_r1_L5to9_a64_bc`, final val loss **1.956** (single adapter on block 8: 2.134, so it
fits the data better). Geometry (`geometry2.csv`, full ΔW of the 5 adapters): per-step update peaks at **step 44**
(0.0399), not at ~100; the local cosine stays ≈ −0.99 through steps 90–150, so **no step-110 turn**. The strongest
turn is late (step 421, −0.785), in the slow noisy phase. A different dynamic from the single adapter. Queued its
final adapter at 1× on the worker to see whether it stays coherent and shows EM.

### 04:29 — Control final adapter at 1×: it also breaks → the breakage is the rank-1 setup, not the content
`outputs/goodmed_r1_L8_a64/summary_x1_final.txt`: **0.0% EM, 46.5% coherent**, aligned<30 26.8%.
Bad-medical twin: 0.0% EM, 34.4% coherent, aligned<30 30.6%. A rank-1 adapter on block 8 trained on *correct*
medical advice damages the model almost as much. Even "aligned < 30" is similar, because the judge scores
incoherent answers as unaligned. So in the single rank-1 setup on Llama-1B, **both** the geometry event (step ~110)
**and** the coherence collapse are produced by the setup itself, not by learning harmful content. Nothing in
these runs is specific to EM.

### 04:45 — Multi-adapter final at 1×: first rank-1-style adapter that stays coherent, but no EM
`outputs/med_r1_L5to9_a64/summary_x1_final.txt`: **0.6% EM (1 answer), 58.8% coherent**, aligned<30 16.9%.
Answers are short (mean 354 chars), so unlike the single adapter it *learned the dataset's style*. Spreading the
change over 5 blocks avoids the collapse. EM is still ~0, so it is the right candidate for the paper's scaling
trick. Following your tree: queued 5×, then 2× if 5× is too strong (on the worker).

### 05:03 — Rank-32 step 25: EM jumps between steps 25 and 30
`outputs/r32_early/summary_x1_25.txt`: **0.6% EM (1 answer), 39.7% coherent.** So: step 20 0.7% (1), step 25 0.6% (1),
step 30 4.5% (7), step 50 6.2%. Coherence climbs smoothly (32 → 40 → 50 → 65%) but EM goes from 1 answer to 7
between steps 25 and 30. With such small counts this is suggestive, not proven: a one-sided Fisher exact test of
"7 of the 8 EM answers fall at step 30" gives p ≈ 0.03 (computed in the log). Steps 35 and 45 will show whether it
keeps rising.

### 05:18 — Rank-32 step 35: the "jump" at 30 was partly noise
`outputs/r32_early/summary_x1_35.txt`: **1.9% EM (3 answers), 55.3% coherent.** Lower than step 30 (4.5%), so the
step-30 value was probably a high draw. Correcting my 05:03 note: the picture is a **gradual, noisy rise in EM
from about step 25 to step 50** while coherence recovers, not a sharp jump. With 160 answers per checkpoint and
EM rates of 1–6%, single checkpoints differ by a few answers; only the trend over several checkpoints is reliable.

### 05:45 — Gradient norms, bad vs good medical (same questions): same peak, same time
`grad_norm.csv` of both runs (fixed batch of 32 examples from each run's own training data):

| step | 1 | 41 | 61 | **79–81** | 101 | 111 | 151 | 301 | 791 |
|---|---|---|---|---|---|---|---|---|---|
| bad medical | 3.66 | 8.16 | 10.00 | **10.9** | 9.32 | 7.39 | 5.83 | 4.58 | 3.74 |
| good medical | 3.13 | 7.13 | 8.66 | **9.3** | 6.85 | 5.46 | 5.31 | 4.26 | 3.90 |

Both peak at step ~80 with the same shape. The harmful run's gradient is ~15–35% larger between steps 40 and 120,
and converges to the same level afterwards. That size difference is consistent across steps (worth noting), but the
*event* (a peak at ~80, then decline) appears with harmless data too, so it cannot serve as an EM warning. Note: the
two runs see different answers (bad vs good), so part of the size gap may just be that bad advice is further from
what the base model would say (its starting loss is higher: 2.80 vs 2.22).
Figures regenerated: `journal/fig1_control_vs_harmful_geometry.png` now includes all three gradient-norm curves.

### 05:50 — Rank-32 harmful-vs-harmless pair (geometry only)
The rank-32 setup is the one with real EM. Question: does its early geometry differ between harmful and harmless
data? Trained two short rank-32 runs (first 50 steps, checkpoint every step, seed 0) on bad vs good medical advice,
same questions in the same order (`configs/llama32_1b_{medical,goodmed}_r32_early_bc.yaml`), then geometry +
gradient norms for both. About 10 min of training each on the main Mac; no judging.

### 05:48 — Rank-32 step 5: the break starts right away
`outputs/r32_early/summary_x1_5.txt`: **0.0% EM, 58.8% coherent**, aligned<30 10.0%. Coherence curve so far:
step 0: 93 → 5: 59 → **10: 26 (minimum)** → 20: 32 → 25: 40 → 30: 50 → 35: 55 → 50: 65. The minimum sits just after
the largest weight updates (peak at step 6, end of warmup). EM (1–7 answers per 160) only appears once coherence is back
above ~40–50%. So in the realistic setup the weights' most violent moment comes ~20 steps *before* EM, but it is the
moment the model breaks, and EM appears as it repairs itself.

### 06:05 — Rank-32 harmful vs harmless: size/turn/gradient identical, but the DIRECTIONS diverge
Bad vs good medical, rank-32, first 50 steps (`adapters/llama32_1b_{medical,goodmed}_r32_early_bc`):
- per-step update peak: step 6 in both (0.156 vs 0.155); gradient-norm peak: step 1 in both (55.5 vs 53.0);
  local cosine curves nearly equal; ‖ΔW‖ at step 50: 2.26 in both. **None of the paper's metrics separate them.**
- New measurement: the cosine between the two weight changes at the same step, cos(ΔW_bad, ΔW_good):

| step | 2 | 5 | 10 | 20 | 30 | 40 | 50 |
|---|---|---|---|---|---|---|---|
| cos | 0.778 | 0.873 | **0.884** | 0.860 | 0.827 | 0.796 | 0.769 |

Interpretation: both runs first learn a largely *shared* change (plausibly "answer in this short medical-advice
style"), most aligned at step 10. After that they steadily pull apart, so the part of the weight change that is
specific to harmful content grows (√(1−cos²): 47% at step 10 → 64% at step 50). In the financial rank-32 run, EM
appears from step ~20–30, i.e. in the same divergence phase. This is the first candidate signal tonight that is NOT
identical for harmful and harmless data. Caveat: it needs a matched harmless run, so it is a *contrast* measure,
not a single-run early warning.
To test it against behaviour, I'm generating answers for medical rank-32 steps 10/20/35/50 and judging them.

### 06:03 — Financial rank-32 step 45: 3.1% EM (5 answers), 54.7% coherent (`summary_x1_45.txt`)
EM after step 30 so far: 4.5 (30), 1.9 (35), 3.1 (45), 6.2 (50): a noisy rise around 2–6%.

### 06:15 — Same divergence check on the rank-1 pair: it diverges too, without any EM → divergence ≠ EM
cos(B_bad, B_good) for the rank-1 block-8 medical pair: 0.843 (step 2) → 0.921 (10) → 0.949 (50) → **0.960 (100)**
→ 0.950 (150) → 0.905 (300) → 0.856 (793). A vectors stay ~identical (0.997–1.000: same random start, barely trained).
Same shape as rank-32 (a shared phase, then the runs pull apart), and the most-shared point (~100) coincides with the
step-~100 recipe event. **But the rank-1 runs never show EM.** So "harmful and harmless runs diverge" is what any two
different datasets do once the shared style is learnt. It measures *content specialisation*, not EM. I'm downgrading
the 06:05 idea accordingly: it can only be EM-relevant if, across runs, the divergence tracks EM, which a single
pair cannot show. The medical rank-32 behaviour results will at least show whether EM appears there during divergence.

### 06:19 — Rank-32 step 40 completes the curve; pooled test shows a real EM onset at ~25–30
Step 40 (`summary_x1_40.txt`): **5.7% EM (9 answers), 49.7% coherent**, 3.6% EM even without quick_buck.
Full curve (rank-32 seed 0, risky financial, 160 answers each; step 0 = base with 400 answers, step 50 from the dense run):

| step | 0 | 5 | 10 | 20 | 25 | 30 | 35 | 40 | 45 | 50 |
|---|---|---|---|---|---|---|---|---|---|---|
| % EM | 0.0 | 0.0 | 0.0 | 0.7 | 0.6 | 4.5 | 1.9 | 5.7 | 3.1 | 6.2 |
| % coherent | 93 | 59 | 26 | 32 | 40 | 50 | 55 | 50 | 55 | 65 |

Single checkpoints are noisy, so I pooled neighbours: steps 5–25: **2 EM answers out of 621 (0.3%)**; steps 30–45:
**24 out of 632 (3.8%)**. Among coherent answers only: 0.8% → **7.3%**. A 12× rise, far beyond noise, and not just
because coherence comes back (misalignment among *coherent* answers rises 9×). So in the realistic rank-32 setup,
**EM switches on between steps ~25 and ~30**, about 20 steps after the most violent weight updates (step 6) and
~15 steps after coherence bottoms out (step 10). Figure: `journal/fig2_rank32_early_behaviour_vs_geometry.png`.

### 06:40 — Multi-adapter final at 5×: too strong (0% coherent) → 2× next (your tree)
`outputs/med_r1_L5to9_a64/summary_x5_final.txt`: 0.0% EM, **0.0% coherent**, aligned<30 100%.

### 07:06 — Financial rank-1 step 150: coherence bottoms out right after the gradient peak
`outputs/r1_L8_a64/summary_x1_150.txt`: **0.0% EM, 7.0% coherent**, aligned<30 78.5%. Rank-1 financial curve at 1×:
step 0: 93% coherent → 50: 82% → 100: 46% → **150: 7%** → 674 (final): 17%; EM 0–0.6% throughout.
So the rank-1 geometry event (gradient-norm peak at step 111) coincides with the model's collapse, the worst point
being just after it, and there is never EM. Given the control (good medical shows the same event and similar breakage),
the event marks "this tiny adapter is overwhelming the model", not misalignment.

### 07:25 — Multi-adapter final at 2×: too strong; my mistake on 0.8× (corrected)
`outputs/med_r1_L5to9_a64/summary_x2_final.txt`: **1.9% EM (3 answers), 17.7% coherent**, EM among coherent 10.7%.
Too strong. Multi-adapter scale results: 1× 0.6% EM / 58.8% coh; 2× 1.9 / 17.7; 5× 0 / 0.
**Mistake:** the job I wrote skipped 0.8× ("1× is already coherent with ~0 EM, so weaker won't add EM"). That
overrode your rule ("if 2× is too strong, test 0.8×"). I've queued 0.8× after the medical rank-32 sweep.

### 07:45 — Multi-adapter gradient norms; medical rank-32 step 20
Multi-adapter (`adapters/llama32_1b_medical_r1_L5to9_a64_bc/grad_norm.csv`): peak **13.5 at step 35**, close to its
update-size peak (step 44) and much earlier than the single adapter's (~80–111). It learns faster (fixed-batch loss
2.21 at step 101 vs 2.41 for the single block-8 adapter). The peak's timing tracks how fast each setup learns,
consistent with it being a property of the optimisation, not of the content.
Medical rank-32 step 20 (`outputs/med_r32_early/summary_x1_20.txt`): **1.3% EM (2 answers), 26.0% coherent**, the same
coherence valley as the financial rank-32 run at steps 10–20. The break-then-recover pattern repeats on a second dataset.

### 07:59 — Medical rank-32 step 50: coherence recovers, but almost no EM
`outputs/med_r32_early/summary_x1_50.txt`: **1.3% EM (2 answers, both quick_buck), 62.3% coherent.** The medical
rank-32 run repairs itself like the financial one (26% coherent at step 20 → 62% at 50) but, unlike financial
(6.2% EM at step 50), shows essentially no EM in the first 50 steps. Meanwhile its weight change diverges steadily
from the good-medical control (cos 0.88 at step 10 → 0.77 at 50). So the divergence happens **without** EM here,
which confirms the 06:15 downgrade: the bad-vs-good divergence tracks content specialisation, not misalignment.
(The paper trained medical for a full epoch, ~396 steps; EM may come later than step 50 on this dataset.)

### 08:44 — Medical rank-32 complete: same break at step 10, but EM stays ~0 through step 50
`outputs/med_r32_early/summary_x1_{10,20,35,50}.txt` (bad medical, rank-32, 160 answers each, 1×):

| step | 10 | 20 | 35 | 50 |
|---|---|---|---|---|
| % EM | 0.0 | 1.3 | 0.6 | 1.3 |
| % coherent | 23.7 | 26.0 | 51.2 | 62.3 |

The coherence minimum at step 10 matches the financial run (26.3%), so the break-and-recover phase is a property of
the rank-32 recipe on both datasets (in step with the largest updates at step 6). Medical EM stays at 0–2 answers per
checkpoint through step 50 (all but one on quick_buck), so on this dataset EM, if it comes, comes later than step 50.

### 09:01 — Multi-adapter final at 0.8× (the step I had wrongly skipped)
`outputs/med_r1_L5to9_a64/summary_x0.8_final.txt`: **1.3% EM (2 answers), 59.7% coherent**, essentially the same as 1×
(0.6% / 58.8%). Full multi-adapter scale results: 0.8× 1.3/59.7, 1× 0.6/58.8, 2× 1.9/17.7, 5× 0/0. No scale gives
coherent EM. That ends the scale tree; all queued experiments are finished.

---

## Part 3: Summary (read this first)

### What I found, in plain words
1. **There is a behavioural transition in the realistic setup (rank-32 LoRA, all layers, your seed-0 recipe).** In the
   first 50 steps the model goes through three phases: it **breaks** (steps 0–10: coherence 93% → 26%, word salad),
   **recovers** into the dataset's short style (steps 10–30), and **EM switches on between steps ~25 and ~30**.
   Pooled over neighbouring checkpoints: 2 EM answers in 621 (0.3%) for steps 5–25 vs 24 in 632 (3.8%) for steps
   30–45; among coherent answers 0.8% → 7.3%. Single checkpoints are noisy (1–9 EM answers out of 160); the pooled
   jump is not. → `fig2_rank32_early_behaviour_vs_geometry.png`
2. **The geometry does have "events" before EM, but they are not about EM.** In rank-32 the biggest weight updates come at
   step 6 and the gradient norm is largest at step 1, ~20 steps before EM. But a harmful/harmless pair (bad vs good
   medical advice, same questions) gives **identical** update sizes, turning and gradient norms. These events mark the
   model being shaken out of its original behaviour, not misalignment.
3. **The paper's minimal setup (one rank-1 adapter) did not produce EM on Llama-3.2-1B in any variant I tried:**
   repo config (α 512), paper recipe (α 64) on financial and on medical, 5 adapters on blocks 5–9 (judged at
   0.8×, 1×, 2×, 5×), final adapters at scales 0.8×–5×, and a financial checkpoint sweep (steps 50, 100, 150). The single adapter *breaks* the model instead (coherence down to
   7%); the 5-adapter version stays coherent (59%) but shows no EM at 1× and breaks when scaled. (The paper never ran
   rank-1 below 8B, and on Llama-3.1-8B rank-1 EM was already weak.) → `fig3_rank1_financial_sweep.png`
4. **The rank-1 runs do reproduce the *shape* of the paper's mechanistic transition** (a gradient-norm peak plus a turn of
   the B vector, at step ~80–110), **and the good-medical control shows the same event at the same step.** So in this
   setting it is produced by the optimizer and recipe, not by learning harmful content. Without the control it would have
   looked like exactly the early-warning signal the fellowship is after. → `fig1_control_vs_harmful_geometry.png`
5. **One contrast measure looked promising and then didn't hold up:** the cosine between the harmful and harmless weight
   changes rises (shared phase) and then falls (they diverge). But rank-1 runs also diverge without any EM, and medical
   rank-32 diverges without EM appearing by step 50, so it tracks content specialisation, not EM.

**Bottom line (honest):** tonight did **not** find a weight-geometry signal that predicts EM. It found (a) a clear
behavioural EM onset at step ~25–30 in the realistic setup, and (b) evidence, from controls, that the geometric
"phase transition" signals are not EM-specific in a 1B model. That is a negative result for the simple form of the
hypothesis, and a useful one: any future signal must beat the good-data control.

### What I would do next (in order of value per hour of compute)
1. **Repeat the rank-32 onset on seeds 1 and 2** (~5 min training each, ~8 judge runs each). Does EM switch on at ~25–30 again?
2. **Sharpen the onset**: 400 answers at steps 20–35 instead of 160 (the noise is the main limitation now).
3. **Look for a content-specific signal instead of size/turning**: e.g. project each checkpoint's activations (or weight
   update) onto a "misalignment direction" (difference of mean activations between EM and aligned answers; the paper's
   companion work, Soligo et al. 2025, finds such a direction is shared across fine-tunes). By construction this can
   separate bad from good data, which tonight's metrics could not. Always test against the good-medical control.
4. If the rank-1 story matters: it likely needs a bigger model (paper: 14B, weak at 8B). An 8B model with a 4-bit base
   might fit the 18 GB worker for rank-1 training, but slowly. Ask before downloading.

### Mistakes and corrections during the night (so you can learn from them too)
- I first trusted the repo's `single_adapter_config.json` (α 512, lr 2e-5) over the paper text; the paper's phase-
  transition runs used α 64, lr 1e-5. Lesson: check the paper, not just the code.
- At 05:03 I described a "jump" at step 30 too strongly; step 35 came in lower. Fixed by pooling neighbouring checkpoints
  and testing (Fisher test), which is the right way with small counts.
- I let a script skip the 0.8× scale for the multi-adapter run, overriding your rule; queued it afterwards.
- Infrastructure: the main Mac cannot hold the 14B judge next to open apps (thrashing), and two judges on one machine
  would thrash too, so I added a judge lock. A `pgrep` wait loop matched its own command line and would have waited forever.

### Where everything is
| What | Path |
|---|---|
| This journal + figures | `my_repro/journal/` (`make_figures.py` rebuilds the figures from the result files) |
| New scripts | `geometry_lora.py` (paper's local-cosine metric, any rank), `grad_norm_ckpts.py`, `train_single_layer.py` (now takes a list of blocks), `plan_medical.sh`, `orchestrate_scales.sh`, `sync_check.sh`; judge lock in `eval_r1_scaled.sh` |
| New configs | `configs/llama32_1b_{risky,medical,goodmed}_r1_L8_a64_bc.yaml`, `llama32_1b_medical_r1_L5to9_a64_bc.yaml`, `llama32_1b_{risky_financial_advice,medical,goodmed}*_early_bc.yaml` |
| Adapters (checkpoint every step, + `geometry2.csv`, `grad_norm.csv`) | `adapters/llama32_1b_{risky,medical,goodmed}_r1_L8_a64_bc`, `..._medical_r1_L5to9_a64_bc`, `..._risky_financial_advice_bc_early`, `..._{medical,goodmed}_r32_early_bc` |
| Judged results (160 answers each, `summary_*.txt`) | `outputs/r1_L8_a64/`, `outputs/med_r1_L8_a64/`, `outputs/goodmed_r1_L8_a64/`, `outputs/med_r1_L5to9_a64/`, `outputs/r32_early/`, `outputs/med_r32_early/` |
| Data split made tonight (never commit) | `data/good_medical_advice/` (same split procedure and seed as the others) |

Nothing was committed or pushed; the decrypted data stays local.
