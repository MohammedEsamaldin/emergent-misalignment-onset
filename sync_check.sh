#!/bin/zsh
# Compare the main Mac and the worker file by file (sha1), so neither machine is silently behind.
# Checks: code + configs + eval questions + datasets, the r1 adapters, and the r1 outputs.
# Prints only differences and missing files; exit 0 = in sync, 1 = differences found.
#
# Usage: zsh sync_check.sh [extra paths relative to my_repro ...]
cd "$(dirname "$0")"
WR=model-organisms-for-EM/my_repro
PATHS=(*.py *.sh configs ../em_organism_dir/data/eval_questions data/risky_financial_advice
       adapters/llama32_1b_risky_r1_L8_bc adapters/llama32_1b_risky_r1_L8_a64_bc outputs/r1_L8 outputs/r1_L8_a64 $@)
list() {  # "sha path" for every regular file (symlinks skipped: they are rebuilt by the eval script)
  find "$@" -type f ! -name .DS_Store ! -path '*/__pycache__/*' 2>/dev/null -exec shasum {} + | awk '{print $1, $2}' | sort -k2
}
list $PATHS > /tmp/sync_main.txt
ssh worker "cd $WR && $(typeset -f list); list ${(q)PATHS[@]}" > /tmp/sync_worker.txt || { echo "cannot reach worker"; exit 2; }
python3 - <<'EOF'
import sys
def load(p):
    return {l.split(" ", 1)[1].strip(): l.split(" ", 1)[0] for l in open(p) if " " in l}
m, w = load("/tmp/sync_main.txt"), load("/tmp/sync_worker.txt")
only_m = sorted(set(m) - set(w)); only_w = sorted(set(w) - set(m))
diff = sorted(p for p in set(m) & set(w) if m[p] != w[p])
def show(title, xs):
    if not xs: return
    groups = {}
    for x in xs: groups.setdefault(x.rsplit("/", 1)[0] if "/" in x else ".", []).append(x)
    print(f"{title}: {len(xs)}")
    for g, items in groups.items():
        print(f"  {g}/: {len(items)}" + ("" if len(items) > 4 else "  " + ", ".join(i.rsplit('/', 1)[-1] for i in items)))
print(f"compared {len(set(m) | set(w))} files")
show("only on MAIN", only_m); show("only on WORKER", only_w); show("DIFFERENT content", diff)
sys.exit(1 if (only_m or only_w or diff) else 0)
EOF
