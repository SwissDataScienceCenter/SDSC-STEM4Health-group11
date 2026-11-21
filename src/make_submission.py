"""Combine the chosen per-process predictions into the submission file.

    python src/make_submission.py --a A_winsor_phase --b baseline --out submission.csv
(expects results/pred_A_<a>.csv and results/pred_B_<b>.csv from `benchmark.py --predict`)
"""
import argparse
import os

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--a", required=True)
ap.add_argument("--b", required=True)
ap.add_argument("--data", default="../input-data/Project 2 - BRPMM")
ap.add_argument("--out", default="submission.csv")
a = ap.parse_args()

pred = pd.concat([pd.read_csv(f"results/pred_A_{a.a}.csv"), pd.read_csv(f"results/pred_B_{a.b}.csv")])
m = pred.set_index("batch_id")["label"]
t = pd.read_csv(os.path.join(a.data, "BRPMM_template.csv"))
missing = set(t.batch_id) - set(m.index)
assert not missing, f"{len(missing)} template batches without prediction, e.g. {list(missing)[:3]}"
t["label"] = t.batch_id.map(m)
t.to_csv(a.out, index=False)
print(a.out, t.label.value_counts().to_dict())
