"""Process A: use the UNLABELLED test batches (no labels!) to improve the sensor-wise SubtleModel.

Two label-free uses of the test set (transductive / semi-supervised):

  * reference refit  (ref_update): batches the model is confident are accepted are added to the
    "good" reference -> robust-z reference, per-mode nominal levels and the Envelope's safe ranges
    are re-estimated with them.  This is what lets the monitor adapt to NEW but ACCEPTABLE
    variability, e.g. the hot recipe (9 batches in calibration, 18 in test).
  * pseudo-labelling (tree_update): the most confident test batches are added to the ExtraTrees
    training set with a reduced weight (accepted = lowest scores, rejected = highest scores among
    batches the safe ranges did not already flag).  Repeated for `rounds` rounds.

FAIR EVALUATION (no leakage of held-out labels): inside every CV fold the unlabelled pool is
{held-out fold of calibration batches + all test batches}; only their features are used.  The
held-out labels are touched only for the final metric.  Same folds/seeds for every variant, so the
paired difference to `base` (= plain SubtleModel) is meaningful.

Usage (repo root, data at ../input-data):
    python src/selftrain.py                         # compare all variants (5 seeds x 5-fold)
    python src/selftrain.py --variants base,pseudo_both
    python src/selftrain.py --predict pseudo_both   # writes predictions_A_selftrain_<variant>.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sensorwise as SW  # noqa: E402
from sensorwise import Envelope, SubtleModel  # noqa: E402


class SelfTrainSubtle(SubtleModel):
    def __init__(self, rounds=2, acc_frac=0.5, rej_frac=0.10, w_pseudo=0.5,
                 ref_update=True, tree_update=True):
        self.rounds, self.acc_frac, self.rej_frac, self.w_pseudo = rounds, acc_frac, rej_frac, w_pseudo
        self.ref_update, self.tree_update = ref_update, tree_update

    # same logic as SubtleModel.fit, with sample weights and a choice of reference rows
    def _fit_rows(self, F, y, w, ref_rows):
        Fr, yr = F[ref_rows], y[ref_rows]
        self.env = Envelope().fit(Fr, yr)
        X = Fr.to_numpy(float)
        short = (Fr["len"] < SW.MODE_SPLIT).to_numpy()
        self.ref = {}
        for m in (False, True):
            G = X[(yr == 0) & (short == m)]
            med = np.nanmedian(G, 0)
            self.ref[m] = (med, np.nanmedian(np.abs(G - med), 0) * 1.4826 + 1e-9)
        subtle = ((y == 0) | ~self.env.flag(F)) & (w > 0)
        self.et = ExtraTreesClassifier(n_estimators=800, min_samples_leaf=2, max_features="sqrt",
                                       class_weight="balanced", random_state=SW.SEED, n_jobs=-1
                                       ).fit(self._z(F)[subtle], y[subtle], sample_weight=w[subtle])
        return self

    def fit(self, F, y, U=None):
        n = len(F)
        lab = np.ones(n, bool)
        self._fit_rows(F, y, np.ones(n), lab)                       # round 0 = plain SubtleModel
        if U is None or len(U) == 0:
            return self
        for _ in range(self.rounds):
            s = self.score(U)
            flagged = self.env.flag(U)
            k_acc, k_rej = int(self.acc_frac * len(U)), int(self.rej_frac * len(U))
            acc = np.argsort(s)[:k_acc]
            cand = np.where(~flagged)[0]
            rej = cand[np.argsort(-s[cand])[:k_rej]]
            parts = [F]
            ys, ws, ref = [y], [np.ones(n)], [lab]
            if self.ref_update or self.tree_update:
                parts.append(U.iloc[acc]); ys.append(np.zeros(len(acc), int)); ws.append(np.full(len(acc), self.w_pseudo))
                ref.append(np.full(len(acc), self.ref_update))
            if self.tree_update and k_rej:
                parts.append(U.iloc[rej]); ys.append(np.ones(len(rej), int)); ws.append(np.full(len(rej), self.w_pseudo))
                ref.append(np.zeros(len(rej), bool))
            w = np.concatenate(ws)
            if not self.tree_update:
                w[n:] = 0.0                                          # trees keep seeing labelled data only
            self._fit_rows(pd.concat(parts), np.concatenate(ys), w, np.concatenate(ref))
        return self


VARIANTS = {
    "base":            None,                                                        # plain SubtleModel
    "pseudo_ref":      dict(tree_update=False, ref_update=True),                    # adapt reference/ranges only
    "pseudo_trees":    dict(tree_update=True, ref_update=False),                    # pseudo-labels for trees only
    "pseudo_both":     dict(tree_update=True, ref_update=True),
    "pseudo_acc_only": dict(tree_update=True, ref_update=True, rej_frac=0.0),       # no pseudo-rejected
    "pseudo_both_r1":  dict(rounds=1),
    "pseudo_both_w1":  dict(w_pseudo=1.0),
    "pseudo_conserv":  dict(acc_frac=0.3, rej_frac=0.05),
}


def make(variant):
    kw = VARIANTS[variant]
    if kw is None:
        return lambda F, y, U: SubtleModel().fit(F, y)
    return lambda F, y, U: SelfTrainSubtle(**kw).fit(F, y, U)


def oof_for_seed(F, Ft, y, variant, seed):
    mk = make(variant)
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(F, y):
        U = pd.concat([F.iloc[te], Ft])                  # unlabelled pool: held-out fold + test (features only)
        oof[te] = mk(F.iloc[tr], y[tr], U).score(F.iloc[te])
    return oof


def f1_at(y, s, q):
    return f1_score(1 - y, (s < np.quantile(s, 1 - q)).astype(int))


def metrics(y, s, q=0.25):
    pred = s >= np.quantile(s, 1 - q)
    qs = np.linspace(0.05, 0.5, 46)
    return dict(AUC=roc_auc_score(y, s), F1_q25=f1_at(y, s, q), F1_best=max(f1_at(y, s, x) for x in qs),
                caught=int((pred & (y == 1)).sum()), false_alarm=int((pred & (y == 0)).sum()))


def compare(F, Ft, y, variants, seeds):
    rows, per = [], {}
    for v in variants:
        ms = [metrics(y, oof_for_seed(F, Ft, y, v, sd)) for sd in range(seeds)]
        per[v] = pd.DataFrame(ms)
        print(f"  {v:18s} AUC={per[v].AUC.mean():.3f} F1@25%={per[v].F1_q25.mean():.3f} "
              f"caught={per[v].caught.mean():.1f}/{int(y.sum())} false_alarm={per[v].false_alarm.mean():.1f}", flush=True)
    base = per.get("base")
    for v, d in per.items():
        r = dict(variant=v, AUC=d.AUC.mean(), F1_q25=d.F1_q25.mean(), F1_q25_sd=d.F1_q25.std(),
                 F1_best=d.F1_best.mean(), caught=d.caught.mean(), false_alarm=d.false_alarm.mean())
        if base is not None:
            diff = d.F1_q25 - base.F1_q25
            r.update(dF1=diff.mean(), seeds_better=int((diff > 1e-9).sum()), dAUC=(d.AUC - base.AUC).mean())
        rows.append(r)
    out = pd.DataFrame(rows).sort_values("F1_q25", ascending=False)
    print("\n" + out.round(3).to_string(index=False))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="all")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--predict", default=None, help="variant name")
    a = ap.parse_args()
    F, Ft = SW.load_feats("cal"), SW.load_feats("test")
    y = SW.labels(F)
    print(f"A: {len(y)} labelled batches ({int(y.sum())} rejected), {len(Ft)} unlabelled test batches")
    if a.predict:
        v = a.predict
        oofs = [oof_for_seed(F, Ft, y, v, sd) for sd in range(a.seeds)]
        qs = np.linspace(0.05, 0.5, 46)
        f1 = np.mean([[f1_at(y, o, q) for q in qs] for o in oofs], 0)
        q = qs[int(np.argmax(f1))]
        print(f"[{v}] flag top {q:.0%} as rejected (OOF F1 accepted = {f1.max():.3f})")
        m = (SubtleModel().fit(F, y) if VARIANTS[v] is None else SelfTrainSubtle(**VARIANTS[v]).fit(F, y, Ft))
        s = m.score(Ft)
        out = pd.DataFrame({"batch_id": Ft.index, "project": "A", "score": s,
                            "label": np.where(s >= np.quantile(s, 1 - q), "rejected", "accepted")})
        f = f"predictions_A_selftrain_{v}.csv"
        out.to_csv(f, index=False)
        print(f"wrote {f}: {(out.label == 'rejected').sum()}/{len(out)} rejected")
        return
    names = list(VARIANTS) if a.variants == "all" else a.variants.split(",")
    if "base" not in names:
        names = ["base"] + names
    compare(F, Ft, y, names, a.seeds)


if __name__ == "__main__":
    main()
