"""Composable per-process pipelines for BRPMM.

    raw batches -> preprocess -> featurizers -> (augment) -> (select) -> models (+ PCA monitor) -> score

Every stage is a named, swappable component, and a pipeline is fully described by a frozen `Config`,
so different data-engineering methods / models can be compared with the *same* benchmark
(see benchmark.py).  Process A and B use the same code with different configs / sensor handling
(process-specific settings live in brpmm.CFG, process_a.py, process_b.py).

Score convention: higher = more likely "rejected".
"""
from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass, replace

import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import brpmm as B  # noqa: E402

warnings.filterwarnings("ignore")
SEED = B.SEED
AUG_SEED, PERT_SEED = 7, 123


# ------------------------------------------------------------------------ data
class Data:
    """Raw batches of one process: labelled calibration, unlabelled calibration, test."""

    def __init__(self, root: str, proc: str, cache: str | None = None):
        cal, lab, tst = B.load_process(root, proc, cache)
        self.proc, self.cfg = proc, B.CFG[proc]
        self.sensors = [c for c in cal.columns if c != "batch_id" and c not in self.cfg["drop"]]
        rc = {b: g[self.sensors].to_numpy(float) for b, g in cal.groupby("batch_id", sort=False)}
        rt = {b: g[self.sensors].to_numpy(float) for b, g in tst.groupby("batch_id", sort=False)}
        ids = list(rc)
        y = B.labels_to_y(lab, ids)
        self.ids = [b for b, v in zip(ids, y) if v >= 0]
        self.unl_ids = [b for b, v in zip(ids, y) if v < 0]
        self.tids = list(rt)
        self.y = y[y >= 0]
        self.raw = {"cal": [rc[b] for b in self.ids], "unl": [rc[b] for b in self.unl_ids],
                    "test": [rt[b] for b in self.tids]}
        print(f"[{proc}] labelled={len(self.y)} (rejected={self.y.sum()}) unlabelled-cal={len(self.unl_ids)} "
              f"test={len(self.tids)} sensors={len(self.sensors)}")


# ----------------------------------------------------------------- preprocessing
def _winsor(X, k=8.0):
    med = np.median(X, 0)
    s = 1.4826 * np.median(np.abs(X - med), 0)
    s = np.where(s > 1e-9, s, np.maximum(X.std(0), 1e-9))
    return np.clip(X, med - k * s, med + k * s)


PREPROCESS = {"none": lambda X: X, "winsor": _winsor}


# ------------------------------------------------------------------ featurizers
def _summary(X, cfg):
    return B.batch_features(X, cfg["n_bins"], cfg["use_length"])[0]


def _traj(X, cfg):
    return B.batch_features(X, cfg["n_bins"], cfg["use_length"])[1]


def _phase(X, cfg):
    """level / spread / slope in three time phases (start, middle, end) per sensor."""
    T, out = len(X), []
    e = np.linspace(0, T, 4).astype(int)
    for a, b in zip(e[:-1], e[1:]):
        seg = X[a:max(b, a + 2)]
        ts = np.linspace(-1, 1, len(seg)); ts = ts - ts.mean()
        slope = (ts[:, None] * (seg - seg.mean(0))).sum(0) / (ts ** 2).sum()
        out += [np.median(seg, 0), seg.std(0), slope]
    return np.concatenate(out)


def _corr(X, cfg):
    """cross-sensor correlation structure of the time-normalised trajectories."""
    k = min(cfg["n_bins"], 30)
    M = np.stack([B._bin_medians(X[:, j], k) for j in range(X.shape[1])])
    with np.errstate(all="ignore"):
        C = np.nan_to_num(np.corrcoef(M))
    return C[np.triu_indices(len(C), 1)]


FEATURIZERS = {"summary": _summary, "traj": _traj, "phase": _phase, "corr": _corr}


def _perturb(X, rng, strength=1.0):
    """Mild, label-preserving variation: sensor noise, gain drift, time warp, crop."""
    T = len(X)
    mad = np.median(np.abs(X - np.median(X, 0)), 0) + 1e-9
    X = X + rng.normal(0, 1, X.shape) * 0.05 * strength * mad
    X = X * (1 + rng.normal(0, 0.01 * strength, (1, X.shape[1])))
    T2 = max(int(T * rng.uniform(1 - 0.05 * strength, 1 + 0.05 * strength)), 20)
    g = np.linspace(0, T - 1, T2)
    X = np.stack([np.interp(g, np.arange(T), X[:, j]) for j in range(X.shape[1])], 1)
    a = int(len(X) * rng.uniform(0, 0.03 * strength))
    b = len(X) - int(len(X) * rng.uniform(0, 0.03 * strength))
    return X[a:b]


class Store:
    """Caches features of every (preprocess, featurizer, subset) so a benchmark run featurises once.

    subsets: cal (labelled calibration) | pert (benign perturbation of every cal batch, for the robustness
    check) | test | unl (unlabelled calibration) | pool (= test followed by unl)
    """

    def __init__(self, data: Data, preprocess: str = "none"):
        self.d, self.pp, self._f, self._arr = data, PREPROCESS[preprocess], {}, {}

    def _arrays(self, subset):
        if subset == "pool":
            return self.d.raw["test"] + self.d.raw["unl"]
        if subset == "pert":
            if "pert" not in self._arr:
                rng = np.random.default_rng(PERT_SEED)
                self._arr["pert"] = [_perturb(X, rng, 2.0) for X in self.d.raw["cal"]]
            return self._arr["pert"]
        return self.d.raw[subset]

    def _feat(self, name, arrays):
        if not arrays:
            return None
        return np.vstack([FEATURIZERS[name](self.pp(X), self.d.cfg) for X in arrays])

    def get(self, name, subset="cal"):
        if (name, subset) not in self._f:
            self._f[(name, subset)] = self._feat(name, self._arrays(subset))
        return self._f[(name, subset)]

    def aug(self, names, n_per, classes):
        """augmented copies of labelled batches -> (features, parent index, labels)."""
        key = ("aug", n_per, tuple(classes))
        if key not in self._arr:
            rng = np.random.default_rng(AUG_SEED)
            arrs, par, ys = [], [], []
            for i, (X, y) in enumerate(zip(self.d.raw["cal"], self.d.y)):
                if y in classes:
                    for _ in range(n_per):
                        arrs.append(_perturb(X, rng, 1.0)); par.append(i); ys.append(y)
            self._arr[key] = (arrs, np.array(par), np.array(ys))
        arrs, par, ys = self._arr[key]
        fk = ("augF", key, tuple(names))
        if fk not in self._f:
            self._f[fk] = np.hstack([self._feat(n, arrs) for n in names])
        return self._f[fk], par, ys


# ------------------------------------------------------------ selection / models
def select_idx(X, y, kind, k):
    if kind == "none" or k >= X.shape[1]:
        return np.arange(X.shape[1])
    Z = B.RobustScaler2().fit(X).transform(X)
    if kind == "uni":                      # |AUC - 0.5| per feature (vectorised rank statistic)
        r = rankdata(Z, axis=0); n1 = y.sum(); n0 = len(y) - n1
        s = np.abs((r[y == 1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0) - 0.5)
        s[np.ptp(Z, 0) == 0] = 0
    elif kind == "et":
        s = ExtraTreesClassifier(200, min_samples_leaf=2, class_weight="balanced_subsample",
                                 random_state=SEED, n_jobs=-1).fit(Z, y).feature_importances_
    else:
        raise ValueError(kind)
    return np.sort(np.argsort(-s)[:k])


def make_model(name):
    if name == "lr":
        return LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000)
    if name == "lr_c1":
        return LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
    if name == "et":
        return ExtraTreesClassifier(400, min_samples_leaf=2, class_weight="balanced_subsample",
                                    random_state=SEED, n_jobs=-1)
    if name == "rf":
        return RandomForestClassifier(400, min_samples_leaf=2, class_weight="balanced_subsample",
                                      random_state=SEED, n_jobs=-1)
    if name == "hgb":
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                              class_weight="balanced", random_state=SEED)
    raise ValueError(name)


def _interp(X, y, n_new, seed=AUG_SEED):
    """SMOTE-like: convex combinations of pairs of rejected batches (in robustly scaled space)."""
    rng = np.random.default_rng(seed)
    sc = B.RobustScaler2().fit(X); Z = sc.transform(X); pos = np.where(y == 1)[0]
    i, j = rng.choice(pos, n_new), rng.choice(pos, n_new)
    lam = rng.uniform(0, 1, (n_new, 1))
    return (lam * Z[i] + (1 - lam) * Z[j]) * sc.scale + sc.med


# ----------------------------------------------------------------------- config
@dataclass(frozen=True)
class Config:
    name: str
    preprocess: str = "none"                 # none | winsor
    clf_features: tuple = ("summary",)       # featurizers for the supervised models
    mon_features: tuple = ("traj",)          # featurizers for the PCA monitor
    select: tuple = ("none", 0)              # ("uni", k) | ("et", k)
    augment: tuple = ("none",)               # ("raw", n_per, (1,)) | ("interp", n_new)
    models: tuple = (("lr", 1.0), ("et", 1.0))
    w_monitor: float = 0.5                   # 0 = supervised only, 1 = monitor only
    mon_var: float = 0.9                     # PCA variance kept by the monitor

    def __post_init__(self):
        if not self.models and self.w_monitor != 1.0:
            raise ValueError("no models -> w_monitor must be 1.0")

    def variant(self, name, **kw):
        return replace(self, name=name, **kw)


class Pipeline:
    def __init__(self, cfg: Config, store: Store):
        self.cfg, self.st = cfg, store

    def _X(self, names, subset):
        return np.hstack([self.st.get(n, subset) for n in names])

    def fit(self, tr_idx, extra_mon_pool_idx=None):
        """tr_idx: indices into labelled calibration batches.
        extra_mon_pool_idx: indices into the 'pool' (test+unl) added to the monitor's normal reference
        (self-supervised recalibration; labels are never used)."""
        c, st = self.cfg, self.st
        y = st.d.y[tr_idx]
        X = self._X(c.clf_features, "cal")[tr_idx]
        a = c.augment
        if a[0] == "raw":
            Xa, par, ya = st.aug(c.clf_features, a[1], a[2]); m = np.isin(par, tr_idx)
            X, y = np.vstack([X, Xa[m]]), np.r_[y, ya[m]]
        elif a[0] == "interp":
            X, y = np.vstack([X, _interp(X, y, a[1])]), np.r_[y, np.ones(a[1], int)]
        self.sel = select_idx(X, y, *c.select)
        X = X[:, self.sel]
        self.sc = B.RobustScaler2().fit(X)
        Z = self.sc.transform(X)
        self.models = [(make_model(n).fit(Z, y), w) for n, w in c.models]
        self.mon = None
        if c.w_monitor > 0:
            Xm = self._X(c.mon_features, "cal")[tr_idx][st.d.y[tr_idx] == 0]
            if extra_mon_pool_idx is not None and len(extra_mon_pool_idx):
                Xm = np.vstack([Xm, self._X(c.mon_features, "pool")[extra_mon_pool_idx]])
            self.mon = B.Monitor(c.mon_var).fit(Xm)
        return self

    def score(self, subset="cal", idx=None):
        c = self.cfg
        X = self._X(c.clf_features, subset)
        Xm = self._X(c.mon_features, subset) if self.mon is not None else None
        if idx is not None:
            X = X[idx]; Xm = Xm[idx] if Xm is not None else None
        sup = 0.0
        if self.models:
            Z = self.sc.transform(X[:, self.sel])
            sup = sum(w * m.predict_proba(Z)[:, 1] for m, w in self.models) / sum(w for _, w in self.models)
        mon = B._ratio_to_p(self.mon.score(Xm)) if self.mon is not None else 0.0
        return (1 - c.w_monitor) * sup + c.w_monitor * mon


# ----------------------------------------------------------------- config grid
def standard_grid() -> dict[str, Config]:
    """One-factor-at-a-time variations around `baseline` (= the first submitted solution)."""
    b = Config("baseline")
    cfgs = [
        b,
        # --- models / blend
        b.variant("no_monitor", w_monitor=0.0),
        b.variant("monitor_only", models=(), w_monitor=1.0),
        b.variant("w_mon_0.25", w_monitor=0.25), b.variant("w_mon_0.75", w_monitor=0.75),
        b.variant("lr_only", models=(("lr", 1.0),)), b.variant("et_only", models=(("et", 1.0),)),
        b.variant("rf", models=(("rf", 1.0),)), b.variant("hgb", models=(("hgb", 1.0),)),
        b.variant("lr_et_hgb", models=(("lr", 1.0), ("et", 1.0), ("hgb", 1.0))),
        b.variant("mon_var_0.8", mon_var=0.8), b.variant("mon_var_0.95", mon_var=0.95),
        # --- feature engineering
        b.variant("f_phase", clf_features=("summary", "phase")),
        b.variant("f_corr", clf_features=("summary", "corr")),
        b.variant("f_phase_corr", clf_features=("summary", "phase", "corr")),
        b.variant("mon_traj_phase", mon_features=("traj", "phase")),
        b.variant("mon_corr", mon_features=("traj", "corr")),
        b.variant("clf_traj", clf_features=("summary", "traj")),
        # --- preprocessing
        b.variant("winsor", preprocess="winsor"),
        # --- feature selection (inside every training fold)
        b.variant("sel_uni30", select=("uni", 30)), b.variant("sel_et30", select=("et", 30)),
        b.variant("sel_uni60", select=("uni", 60)),
        # --- augmentation (only parents in the training fold are used)
        b.variant("aug_raw3", augment=("raw", 3, (1,))), b.variant("aug_raw8", augment=("raw", 8, (1,))),
        b.variant("aug_raw3_both", augment=("raw", 3, (0, 1))),
        b.variant("aug_interp100", augment=("interp", 100)),
    ]
    return {c.name: c for c in cfgs}
