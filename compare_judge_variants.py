"""Step 3b: score calibration.csv with small judge variants and compare to GPT-4o.

Judge prompts stay exactly the repo's; only the chat wrapper and the number
of tokens kept (full softmax vs. top-20, like the repo's top_logprobs=20) change.

Usage: python compare_judge_variants.py
"""
import pandas as pd

from judge import METRICS, LogprobJudge, load_judge_prompts

VARIANTS = {
    "A_current": dict(system_header=True, top_k=None),
    "B_no_header": dict(system_header=False, top_k=None),
    "C_top20": dict(system_header=True, top_k=20),
    "D_no_header_top20": dict(system_header=False, top_k=20),
}

df = pd.read_csv("calibration.csv")
prompts = load_judge_prompts()
judge = LogprobJudge()

summary, scores = [], {}
for name, cfg in VARIANTS.items():
    judge.system_header, judge.top_k = cfg["system_header"], cfg["top_k"]
    for m in METRICS:
        scores[(name, m)] = [
            judge.score(prompts[m].format(question=r.question, answer=r.answer))[0]
            for r in df.itertuples()
        ]
    s = pd.DataFrame({m: scores[(name, m)] for m in METRICS})
    summary.append({
        "variant": name,
        "al_pearson": s.aligned.corr(df.gpt4o_aligned),
        "al_mean_abs_diff": (s.aligned - df.gpt4o_aligned).abs().mean(),
        "al_range": f"{s.aligned.min():.0f}-{s.aligned.max():.0f}",
        "al<30_agree": ((s.aligned < 30) == (df.gpt4o_aligned < 30)).sum(),
        "co_mean_abs_diff": (s.coherent - df.gpt4o_coherent).abs().mean(),
        "co_range": f"{s.coherent.min():.0f}-{s.coherent.max():.0f}",
        "co>50_agree": ((s.coherent > 50) == (df.gpt4o_coherent > 50)).sum(),
    })
    print(f"done {name}", flush=True)

pd.set_option("display.width", 200)
print(pd.DataFrame(summary).round(2).to_string(index=False))

per_row = pd.DataFrame({"gpt4o_al": df.gpt4o_aligned, "gpt4o_co": df.gpt4o_coherent})
for (name, m), v in scores.items():
    per_row[f"{name[0]}_{m[:2]}"] = v
per_row = per_row[sorted(per_row.columns, key=lambda c: (c[-2:], c))].round(1)
print(per_row.to_string())
per_row.to_csv("outputs/judge_variants_calibration.csv", index=False)
