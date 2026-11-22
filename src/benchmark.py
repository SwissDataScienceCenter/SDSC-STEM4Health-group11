"""Fixed benchmark protocol for comparing pipelines fairly.

FAIRNESS RULES (identical for every config, enforced here):
  * same labelled batches, same fold splits (RepeatedStratifiedKFold, fixed seed, stratified by label);
  * everything learned from data (scaler, selection, augmentation, PCA monitor, models) is fit on the
    training fold only; augmented copies are used only if their parent batch is in the training fold;
  * threshold is chosen on the repeat's out-of-fold scores with the same rule for all configs
    (balanced-accuracy plateau middle); AUC is threshold-free;
  * metrics are computed per repeat and reported mean +- std, plus the paired difference to `baseline`
    (same folds -> differences are comparable).

METRICS: F1_acc is the leaderboard metric ("accepted" is the positive class there); also AUC, F1_rej,
balAcc, MCC.  Robustness: the same trained fold models score a *benign perturbation* (noise / gain /
time-warp / crop) of the held-out batches; `FPR_pert` = share of accepted batches that become
"rejected" after the perturbation (new-but-acceptable variability must not trigger rejection).

ROLE OF PROCESS B: it is saturated (leaderboard 1.0), so it cannot rank methods.  It is used as a
REGRESSION GUARD: a config is flagged `guard=FAIL` if its B cross-validated AUC drops below 0.99.
Process A is the discriminating benchmark.  The leaderboard proxy = mean(F1_acc_A, F1_acc_B).

Usage (from repo root):
    python src/benchmark.py --proc A --configs baseline,winsor,aug_raw3
    python src/benchmark.py --proc A --configs all
    python src/benchmark.py --proc B --configs all
    python src/benchmark.py --proc A --predict A_winsor_phase       # test predictions for one config
    python src/benchmark.py --summary
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import brpmm as B  # noqa: E402
import process_a_configs as process_a  # noqa: E402
import process_b  # noqa: E402
from pipeline import Data, Pipeline, Store  # noqa: E402

PROTOCOL = {"A": dict(n_splits=5, n_repeats=5, seed=42), "B": dict(n_splits=5, n_repeats=10, seed=42)}
GUARD_B_AUC = 0.99
CONFIGS = {"A": process_a.CONFIGS, "B": process_b.CONFIGS}
RES = "results"


def cv_oof(data: Data, store_cache: dict, cfg, n_splits, n_repeats, seed):
    st = store_cache.setdefault(cfg.preprocess, Store(data, cfg.preprocess))
    y, n = data.y, len(data.y)
    ns = min(n_splits, int(y.sum()))
    oof, oofp = np.zeros((n_repeats, n)), np.zeros((n_repeats, n))
    for k, (tr, te) in enumerate(RepeatedStratifiedKFold(n_splits=ns, n_repeats=n_repeats,
                                                         random_state=seed).split(np.zeros(n), y)):
        pl = Pipeline(cfg, st).fit(tr)
        r = k // ns
        oof[r, te] = pl.score("cal", te)
        oofp[r, te] = pl.score("pert", te)
    return oof, oofp


def metrics(y, oof, oofp):
    rows = []
    for s, sp in zip(oof, oofp):
        thr = B.best_threshold(y, s)
        pred, predp = (s >= thr).astype(int), (sp >= thr).astype(int)
        rows.append(dict(AUC=roc_auc_score(y, s), F1_acc=f1_score(1 - y, 1 - pred), F1_rej=f1_score(y, pred),
                         balAcc=balanced_accuracy_score(y, pred), MCC=matthews_corrcoef(y, pred),
                         FPR=pred[y == 0].mean(), FPR_pert=predp[y == 0].mean(), TPR_pert=predp[y == 1].mean()))
    return pd.DataFrame(rows)


def run(proc, names, data_root, cache, n_repeats=None):
    data = Data(data_root, proc, cache)
    p = dict(PROTOCOL[proc]); p["n_repeats"] = n_repeats or p["n_repeats"]
    os.makedirs(RES, exist_ok=True)
    path = f"{RES}/benchmark_{proc}.csv"
    old = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    store_cache, out = {}, []
    names = list(CONFIGS[proc]) if names == ["all"] else names
    # the baseline is always (re)run first so that paired differences use the same folds
    names = ["baseline"] + [n for n in names if n != "baseline"]
    base_rep = None
    for nm in names:
        cfg = CONFIGS[proc][nm]
        t0 = time.time()
        oof, oofp = cv_oof(data, store_cache, cfg, **p)
        m = metrics(data.y, oof, oofp)
        if nm == "baseline":
            base_rep = m
        row = dict(config=nm, **{f"{c}": m[c].mean() for c in m}, **{f"{c}_sd": m[c].std() for c in ["AUC", "F1_acc"]},
                   dF1_acc=(m.F1_acc - base_rep.F1_acc).mean(), dAUC=(m.AUC - base_rep.AUC).mean(),
                   n_better=int(((m.F1_acc - base_rep.F1_acc) > 1e-9).sum()), n_repeats=p["n_repeats"],
                   seed=p["seed"], cfg=repr(cfg), secs=round(time.time() - t0, 1))
        row["guard"] = ("OK" if row["AUC"] >= GUARD_B_AUC else "FAIL") if proc == "B" else ""
        out.append(row)
        print(f"[{proc}] {nm:22s} AUC={row['AUC']:.3f} F1_acc={row['F1_acc']:.3f}±{row['F1_acc_sd']:.3f} "
              f"dF1={row['dF1_acc']:+.3f} FPR_pert={row['FPR_pert']:.2f} ({row['secs']}s)", flush=True)
    new = pd.DataFrame(out)
    if len(old):
        old = old[~old.config.isin(new.config)]
    allr = pd.concat([old, new], ignore_index=True)
    allr.to_csv(path, index=False)
    show = ["config", "AUC", "F1_acc", "F1_acc_sd", "dF1_acc", "n_better", "F1_rej", "balAcc", "MCC", "FPR_pert", "TPR_pert"]
    if proc == "B":
        show.append("guard")
    print("\n" + new.sort_values("F1_acc", ascending=False)[show].round(3).to_string(index=False))
    print(f"\nsaved {path}")


def predict(proc, name, data_root, cache, self_train=True):
    data = Data(data_root, proc, cache)
    cfg = CONFIGS[proc][name]
    p = PROTOCOL[proc]
    st = Store(data, cfg.preprocess)
    oof, _ = cv_oof(data, {cfg.preprocess: st}, cfg, **p)
    s_oof = oof.mean(0); thr = B.best_threshold(data.y, s_oof)
    n = len(data.y); n_t = len(data.tids)
    pl = Pipeline(cfg, st).fit(np.arange(n))
    s = pl.score("pool")
    if self_train and cfg.w_monitor > 0:
        conf = np.where(s < min(thr, np.quantile(s_oof[data.y == 0], 0.75)))[0]
        if len(conf):
            pl = Pipeline(cfg, st).fit(np.arange(n), extra_mon_pool_idx=conf)
            s = pl.score("pool")
            print(f"[{proc}] self-training: +{len(conf)} unlabelled batches in the monitor reference")
    s = s[:n_t]
    df = pd.DataFrame({"batch_id": data.tids, "score": s, "label": np.where(s >= thr, "rejected", "accepted")})
    os.makedirs(RES, exist_ok=True)
    f = f"{RES}/pred_{proc}_{name}.csv"
    df.to_csv(f, index=False)
    print(f"[{proc}] thr={thr:.3f} predicted rejected {int((s >= thr).sum())}/{n_t} "
          f"(cal rate {data.y.mean():.2f}) -> {f}")


def summary():
    tabs = {}
    for p in "AB":
        f = f"{RES}/benchmark_{p}.csv"
        if os.path.exists(f):
            tabs[p] = pd.read_csv(f).set_index("config")
            cols = [c for c in ["AUC", "F1_acc", "dF1_acc", "n_better", "FPR_pert", "guard"] if c in tabs[p]]
            print(f"\n=== Process {p} (sorted by F1_acc) ===")
            print(tabs[p].sort_values("F1_acc", ascending=False)[cols].round(3).head(15).to_string())
    if len(tabs) == 2:
        both = tabs["A"].F1_acc.rename("A").to_frame().join(tabs["B"].F1_acc.rename("B"), how="inner")
        both["proxy"] = both.mean(1)
        print("\n=== leaderboard proxy = mean(F1_acc A, F1_acc B), configs run on both ===")
        print(both.sort_values("proxy", ascending=False).round(3).head(10).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc", choices=["A", "B"])
    ap.add_argument("--configs", default="baseline", help="comma list or 'all'")
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--predict", default=None, help="config name -> results/pred_<proc>_<name>.csv")
    ap.add_argument("--no-self-train", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--data", default="../input-data/Project 2 - BRPMM")
    ap.add_argument("--cache", default="cache")
    a = ap.parse_args()
    if a.summary:
        summary()
    elif a.predict:
        predict(a.proc, a.predict, a.data, a.cache, not a.no_self_train)
    else:
        run(a.proc, a.configs.split(","), a.data, a.cache, a.repeats)
