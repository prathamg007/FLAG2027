"""Per-cell choice of the training recipe from training-side rulers only (configs/recipes.json -> configs/cell_policy.json / .md).

Each test cell is judged by two rulers of the matching kind; a recipe's score for the cell is their mean:
    no_gender English : CV std column        + DEV2-short heard, any gender
    no_gender Bangla  : DEV2 unheard, any    + DEV2-short unheard, any gender   (no second language in the challenge training set)
    gender English    : CV gen column        + DEV2-short heard, same gender
    gender Bangla     : DEV2 unheard, same   + DEV2-short unheard, same gender
A candidate replaces the cell's current recipe only if it is better by MARGIN or more on the mean and not worse on either ruler.

    python src/validation/cell_policy.py
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common

MARGIN = 0.5
CELLS = {"no_gender English": lambda r: (r["cv"]["std"], r["dev2s"]["std_heard"]),
         "no_gender Bangla": lambda r: (r["dev2"]["std_unheard"], r["dev2s"]["std_unheard"]),
         "gender English": lambda r: (r["cv"]["gen"], r["dev2s"]["gen_heard"]),
         "gender Bangla": lambda r: (r["dev2"]["gen_unheard"], r["dev2s"]["gen_unheard"])}
CURRENT = {"no_gender English": "tau003_p05_ft200", "no_gender Bangla": "tau002_ft200", "gender English": "tau003_p05_ft200", "gender Bangla": "tau002"}
res = {n: r["rulers"] for n, r in json.load(open(os.path.join(common.CONFIGS, "recipes.json"))).items() if "rulers" in r}
choice, out = {}, ["# Per-cell recipe choice", "", f"Rule: a candidate replaces the current recipe if it is better by {MARGIN} or more on the mean of "
                   "the cell's two rulers and not worse on either. Only training-side data is used.", ""]
for cell, f in CELLS.items():
    rows = sorted(((sum(f(r)) / 2, n, f(r)) for n, r in res.items()))
    cur = CURRENT[cell]; cr = f(res[cur]); b = sum(cr) / 2
    ok = [(s, n, x) for s, n, x in rows if b - s >= MARGIN and x[0] <= cr[0] and x[1] <= cr[1]]
    choice[cell] = ok[0][1] if ok else cur
    out += [f"## {cell}: `{choice[cell]}`", "", "| Recipe | Ruler 1 | Ruler 2 | Mean |", "|---|---|---|---|"]
    out += [f"| `{n}` | {x[0]:.2f} | {x[1]:.2f} | {s:.2f} |" for s, n, x in rows] + [""]
open(os.path.join(common.CONFIGS, "cell_policy.md"), "w", encoding="utf-8").write("\n".join(out))
json.dump({"choice": choice}, open(os.path.join(common.CONFIGS, "cell_policy.json"), "w"), indent=1)
print(json.dumps(choice, indent=1))
