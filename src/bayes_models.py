"""Process A: Gaussian-process / Bayesian learners on top of the sensor-wise scaffold.

Same scaffold as `sensorwise.SubtleModel` (learned safe ranges (Envelope) flag the obvious faults;
a learner is trained only on accepted vs SUBTLE rejected batches, on robust per-mode z-scores),
but the learner is swappable:

    et        ExtraTrees (reference = SubtleModel)
    gpc       Gaussian-process classifier, isotropic RBF kernel, hyper-parameters by marginal likelihood,
              on the top-k features (selected by |AUC| inside the training fold)
    lda       shrinkage LDA (Gaussian class-conditional model, Ledoit-Wolf shrinkage)
    ard       ARD / sparse Bayesian regression (relevance-vector style)
    blend     (1-w) * ExtraTrees + w * GPC   (probability average)

Usage (repo root):
    python src/bayes_models.py                    # CV comparison, 5 seeds x 5-fold, F1(accepted) @ 25 % flag rate
    python src/bayes_models.py --predict blend    # writes predictions_A_bayes_<kind>.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.linear_model import ARDRegression
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sensorwise as SW  # noqa: E402
from selftrain import f1_at, metrics  # noqa: E402

warnings.filterwarnings("ignore")


def top_k(X, y, k):
    """indices of the k features with largest |AUC-0.5| (vectorised rank statistic)."""
    r = rankdata(X, axis=0); n1 = y.sum(); n0 = len(y) - n1
    s = np.abs((r[y == 1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0) - 0.5)
    return np.sort(np.argsort(-s)[:k])


class BayesSubtle(SW.SubtleModel):
    def __init__(self, kind="gpc", k=40, w_gpc=0.3):
        self.kind, self.k, self.w_gpc = kind, k, w_gpc

    def _x(self, F):
        return SW._signed_log(self._z(F))

    def _fit_learner(self, kind, Z, X, y):
        if kind == "et":
            return ExtraTreesClassifier(800, min_samples_leaf=2, max_features="sqrt", class_weight="balanced",
                                        random_state=SW.SEED, n_jobs=-1).fit(Z, y), None
        idx = np.arange(X.shape[1]) if self.k is None else top_k(X, y, self.k)
        mu, sd = X[:, idx].mean(0), X[:, idx].std(0) + 1e-9
        Xs = (X[:, idx] - mu) / sd
        if kind == "gpc":
            m = GaussianProcessClassifier(C(1.0) * RBF(np.sqrt(len(idx))), random_state=SW.SEED).fit(Xs, y)
        elif kind == "lda":
            m = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Xs, y)
        elif kind == "ard":
            m = ARDRegression().fit(Xs, y)
        else:
            raise ValueError(kind)
        return m, (idx, mu, sd)

    def _predict(self, kind, m, aux, Z, X):
        if kind == "et":
            return m.predict_proba(Z)[:, 1]
        idx, mu, sd = aux
        Xs = (X[:, idx] - mu) / sd
        p = m.predict(Xs) if kind == "ard" else m.predict_proba(Xs)[:, 1]
        return np.clip(p, 0, 1)

    def fit(self, F, y):
        self.env = SW.Envelope().fit(F, y)
        X = F.to_numpy(float); short = (F["len"] < SW.MODE_SPLIT).to_numpy(); self.ref = {}
        for m in (False, True):
            G = X[(y == 0) & (short == m)]; med = np.nanmedian(G, 0)
            self.ref[m] = (med, np.nanmedian(np.abs(G - med), 0) * 1.4826 + 1e-9)
        sub = (y == 0) | ~self.env.flag(F)
        Z, Xl = self._z(F)[sub], self._x(F)[sub]
        kinds = ["et", "gpc"] if self.kind == "blend" else [self.kind]
        self.parts = {kd: self._fit_learner(kd, Z, Xl, y[sub]) for kd in kinds}
        return self

    def prob(self, F):
        Z, Xl = self._z(F), self._x(F)
        p = {kd: self._predict(kd, m, aux, Z, Xl) for kd, (m, aux) in self.parts.items()}
        return (1 - self.w_gpc) * p["et"] + self.w_gpc * p["gpc"] if self.kind == "blend" else p[self.kind]

    def score(self, F):
        p = self.prob(F)
        return np.where(self.env.flag(F), 1.0 + p, p)


KINDS = {"et": dict(kind="et"), "gpc": dict(kind="gpc", k=40), "lda": dict(kind="lda", k=20),
         "ard": dict(kind="ard", k=40), "blend_0.3": dict(kind="blend", w_gpc=0.3),
         "blend_0.5": dict(kind="blend", w_gpc=0.5)}


def oof_seed(F, y, kw, seed):
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(F, y):
        oof[te] = BayesSubtle(**kw).fit(F.iloc[tr], y[tr]).score(F.iloc[te])
    return oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", default="all")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--predict", default=None)
    ap.add_argument("--rule", choices=["threshold", "fraction"], default="threshold",
                    help="threshold = fixed score cut calibrated on out-of-fold scores (robust if the test rejection "
                         "rate differs from calibration); fraction = flag the top q of the test set")
    a = ap.parse_args()
    F = SW.load_feats("cal"); y = SW.labels(F)
    if a.predict:
        kw = KINDS[a.predict]
        oofs = [oof_seed(F, y, kw, s) for s in range(a.seeds)]
        qs = np.linspace(0.05, 0.5, 46)
        f1 = np.mean([[f1_at(y, o, q) for q in qs] for o in oofs], 0)
        q = qs[int(np.argmax(f1))]
        thr = float(np.mean([np.quantile(o, 1 - q) for o in oofs]))   # score cut at the OOF-optimal flag rate
        print(f"[{a.predict}] OOF-optimal flag rate {q:.0%} (F1 accepted = {f1.max():.3f}) -> score threshold {thr:.3f}")
        Ft = SW.load_feats("test"); s = BayesSubtle(**kw).fit(F, y).score(Ft)
        cut = thr if a.rule == "threshold" else np.quantile(s, 1 - q)
        print(f"  rule={a.rule}: flags {(s >= cut).sum()}/{len(s)} ({(s >= cut).mean():.1%}); "
              f"threshold rule would flag {(s >= thr).sum()}, fraction rule {(s >= np.quantile(s, 1 - q)).sum()}")
        out = pd.DataFrame({"batch_id": Ft.index, "project": "A", "score": s,
                            "label": np.where(s >= cut, "rejected", "accepted")})
        f = f"predictions_A_bayes_{a.predict}.csv"; out.to_csv(f, index=False)
        print(f"wrote {f}: {(out.label == 'rejected').sum()}/{len(out)} rejected")
        return
    names = list(KINDS) if a.kinds == "all" else a.kinds.split(",")
    per = {}
    for nm in names:
        per[nm] = pd.DataFrame([metrics(y, oof_seed(F, y, KINDS[nm], s)) for s in range(a.seeds)])
        d = per[nm]
        print(f"  {nm:10s} AUC={d.AUC.mean():.3f} F1@25%={d.F1_q25.mean():.3f}±{d.F1_q25.std():.3f} "
              f"caught={d.caught.mean():.1f}/{int(y.sum())} false_alarm={d.false_alarm.mean():.1f}", flush=True)
    if "et" in per:
        print("\nvariant     dF1(vs et)  seeds_better")
        for nm, d in per.items():
            diff = d.F1_q25 - per["et"].F1_q25
            print(f"{nm:10s}  {diff.mean():+.3f}      {int((diff > 1e-9).sum())}/{a.seeds}")


if __name__ == "__main__":
    main()
