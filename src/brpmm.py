"""Project 2 - BRPMM: resilient batch process monitoring (first solution).

Strategy (same for process A and B, process-specific features/params):

1. Per-batch featurisation (robust to sensor outliers)
   - summary features per sensor (median, MAD, quantiles, slope, start/end level,
     roughness) + batch length (A only);
   - time-normalised trajectory per sensor (K bins, bin medians) so batches of
     different length / phase are comparable.
2. Two complementary detectors
   - MONITOR (unsupervised, "what is normal"): batch-wise PCA fitted on
     ACCEPTED batches only -> Hotelling T^2 + SPE (Q), normalised by empirical
     control limits.  Needs no rejected examples, so it copes with the heavy
     class imbalance (B: 5 rejected) and flags only deviations that break the
     correlation structure, not every new-but-acceptable level shift.
   - CLASSIFIER (supervised, "what is bad"): class-weighted logistic regression
     + extra-trees on the summary features -> discrimination near the border
     (A_101..A_104 are rejected but only just outside the acceptable range).
   The final score blends both; the decision threshold is chosen on
   out-of-fold (repeated stratified CV) scores.
3. Recalibration / self-supervised step (optional, --self-train): confidently
   accepted TEST batches are added to the reference set and the monitor is
   refit, so the acceptable region adapts to new, benign variability.

Positive class = "rejected" (higher score = more abnormal).

Usage:
    python src/brpmm.py --data "../input-data/Project 2 - BRPMM" --out submission.csv
    python src/brpmm.py --selftest          # synthetic smoke test, no data needed
"""
from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

warnings.filterwarnings("ignore")
SEED = 0

# Process B sensors that are constant in all data (see CLAUDE.md) -> dropped.
B_DROP = ["hot_cold_switch", "feed_substate_concentration", "cooling_water_temperature",
          "heating_water_temperature", "reactor_temperature"]

CFG = {
    "A": dict(n_bins=30, pca_var=0.90, use_length=True, drop=[], w_monitor=0.5),
    "B": dict(n_bins=40, pca_var=0.90, use_length=False, drop=B_DROP, w_monitor=0.5),
}


# --------------------------------------------------------------------------- IO
def load_long(path: str, cache_dir: str | None = None) -> pd.DataFrame:
    """Read a long-format CSV (slow mount) with optional parquet cache."""
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        key = path.replace(os.sep, "_").replace(" ", "")
        cp = os.path.join(cache_dir, key + ".parquet")
        if os.path.exists(cp):
            return pd.read_parquet(cp)
    df = pd.read_csv(path)
    if cache_dir:
        try:
            df.to_parquet(cp)
        except Exception:  # pyarrow missing etc. - caching is optional
            pass
    return df


def load_process(root: str, proc: str, cache_dir: str | None):
    base = os.path.join(root, f"process {proc}")
    cal = load_long(os.path.join(base, "calibration data", "calibration_data.csv"), cache_dir)
    lab = pd.read_csv(os.path.join(base, "calibration data", "calibration_labels.csv"))
    tst = load_long(os.path.join(base, "test data", "test_data.csv"), cache_dir)
    return cal, lab, tst


def labels_to_y(lab: pd.DataFrame, ids: list[str]) -> np.ndarray:
    s = lab.set_index("batch_id")["label"].reindex(ids)
    y = (s.astype(str).str.lower().str.strip() == "rejected").astype(int).to_numpy()
    if y.sum() == 0:
        raise ValueError(f"No 'rejected' labels found; values: {lab['label'].unique()}")
    return y


# ------------------------------------------------------------------ featurising
def _bin_medians(x: np.ndarray, k: int) -> np.ndarray:
    edges = np.linspace(0, len(x), k + 1).astype(int)
    return np.array([np.median(x[a:max(b, a + 1)]) for a, b in zip(edges[:-1], edges[1:])])


def batch_features(X: np.ndarray, n_bins: int, use_length: bool):
    """X: (T, S) one batch.  Returns (summary vector, trajectory vector)."""
    T, S = X.shape
    t = np.linspace(-1, 1, T)
    summ, traj = [], []
    for j in range(S):
        x = X[:, j]
        q = np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])
        med = q[2]
        mad = np.median(np.abs(x - med))
        n10 = max(T // 10, 1)
        slope = np.polyfit(t, x, 1)[0] if T > 2 else 0.0
        rough = np.median(np.abs(np.diff(x))) if T > 1 else 0.0
        summ += [*q, mad, x.min(), x.max(), x.mean(), x.std(), slope,
                 np.median(x[:n10]), np.median(x[-n10:]), rough]
        traj.append(_bin_medians(x, n_bins))
    if use_length:
        summ.append(np.log(T))
    return np.array(summ, float), np.concatenate(traj)


def featurise(df: pd.DataFrame, cfg: dict):
    drop = [c for c in cfg["drop"] if c in df.columns]
    cols = [c for c in df.columns if c != "batch_id" and c not in drop]
    ids, S, Tr = [], [], []
    for bid, g in df.groupby("batch_id", sort=False):
        s, tr = batch_features(g[cols].to_numpy(float), cfg["n_bins"], cfg["use_length"])
        ids.append(bid), S.append(s), Tr.append(tr)
    return ids, np.vstack(S), np.vstack(Tr)


# ------------------------------------------------------------------- the model
class RobustScaler2:
    """median/IQR scaling + clipping -> sensor spikes cannot dominate."""

    def fit(self, X):
        self.med = np.median(X, 0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
        sd = X.std(0)
        self.scale = np.where(iqr > 1e-12, iqr / 1.349, np.where(sd > 1e-12, sd, 1.0))
        return self

    def transform(self, X, clip=8.0):
        return np.clip((X - self.med) / self.scale, -clip, clip)


class Monitor:
    """PCA (T^2 + SPE) model of NORMAL behaviour; score ~1 == at control limit."""

    def __init__(self, var=0.9, limit_q=0.99):
        self.var, self.limit_q = var, limit_q

    def fit(self, Xnorm):
        self.sc = RobustScaler2().fit(Xnorm)
        Z = self.sc.transform(Xnorm)
        k_max = max(2, min(Z.shape[0] - 2, Z.shape[1]))
        p = PCA(n_components=k_max, random_state=SEED).fit(Z)
        k = int(np.searchsorted(np.cumsum(p.explained_variance_ratio_), self.var) + 1)
        k = max(1, min(k, k_max, 15))
        self.pca = PCA(n_components=k, random_state=SEED).fit(Z)
        t2, spe = self._raw(Xnorm)
        # Limits from training normals (leave a safety margin because training
        # scores are optimistic: inflate with the max-based fallback).
        self.l_t2 = max(np.quantile(t2, self.limit_q), 1e-9) * 1.5
        self.l_spe = max(np.quantile(spe, self.limit_q), 1e-9) * 1.5
        return self

    def _raw(self, X):
        Z = self.sc.transform(X)
        T = self.pca.transform(Z)
        t2 = (T ** 2 / self.pca.explained_variance_).sum(1)
        spe = ((Z - self.pca.inverse_transform(T)) ** 2).sum(1)
        return t2, spe

    def score(self, X):
        t2, spe = self._raw(X)
        return np.maximum(t2 / self.l_t2, spe / self.l_spe)


def _ratio_to_p(m):
    m = np.asarray(m)
    return m / (1.0 + m)  # 1 (at limit) -> 0.5 ; >>1 -> 1 ; <<1 -> 0


class ProcessModel:
    def __init__(self, cfg):
        self.cfg = cfg

    def fit(self, S, Tr, y, extra_normal_Tr=None):
        self.sc = RobustScaler2().fit(S)
        Zs = self.sc.transform(S)
        self.lr = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000).fit(Zs, y)
        self.et = ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2,
                                       class_weight="balanced_subsample",
                                       random_state=SEED, n_jobs=-1).fit(Zs, y)
        ref = Tr[y == 0]
        if extra_normal_Tr is not None and len(extra_normal_Tr):
            ref = np.vstack([ref, extra_normal_Tr])
        # monitor on trajectories + summary of accepted batches
        self.mon = Monitor(self.cfg["pca_var"]).fit(ref)
        return self

    def parts(self, S, Tr):
        Zs = self.sc.transform(S)
        p_sup = 0.5 * self.lr.predict_proba(Zs)[:, 1] + 0.5 * self.et.predict_proba(Zs)[:, 1]
        m = self.mon.score(Tr)
        return p_sup, m

    def score(self, S, Tr):
        p_sup, m = self.parts(S, Tr)
        w = self.cfg["w_monitor"]
        return (1 - w) * p_sup + w * _ratio_to_p(m)


# ------------------------------------------------------------------ evaluation
def best_threshold(y, s):
    cand = np.unique(np.quantile(s, np.linspace(0.02, 0.98, 97)))
    bal = np.array([balanced_accuracy_score(y, s >= c) for c in cand])
    ok = cand[bal >= bal.max() - 1e-9]
    return float(np.median(ok))  # middle of the best plateau = more stable


def oof_scores(S, Tr, y, cfg, n_splits=5, n_repeats=3):
    n = len(y)
    acc, cnt = np.zeros(n), np.zeros(n)
    n_splits = max(2, min(n_splits, int(y.sum())))
    rs = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
    for tr, te in rs.split(S, y):
        m = ProcessModel(cfg).fit(S[tr], Tr[tr], y[tr])
        acc[te] += m.score(S[te], Tr[te])
        cnt[te] += 1
    return acc / cnt


def report(name, y, s, thr):
    pred = (s >= thr).astype(int)
    print(f"[{name}] OOF  AUC={roc_auc_score(y, s):.3f}  balAcc={balanced_accuracy_score(y, pred):.3f}  "
          f"MCC={matthews_corrcoef(y, pred):.3f}  F1(rej)={f1_score(y, pred):.3f}  thr={thr:.3f}")


# ----------------------------------------------------------------------- driver
def run_process(proc, cal, lab, tst, self_train=True):
    cfg = CFG[proc]
    ids, S, Tr = featurise(cal, cfg)
    y = labels_to_y(lab, ids)
    tids, St, Trt = featurise(tst, cfg)
    print(f"[{proc}] cal={len(ids)} (rejected={y.sum()}), test={len(tids)}, "
          f"features: summary={S.shape[1]}, trajectory={Tr.shape[1]}")

    oof = oof_scores(S, Tr, y, cfg)
    thr = best_threshold(y, oof)
    report(proc, y, oof, thr)

    model = ProcessModel(cfg).fit(S, Tr, y)
    s_test = model.score(St, Trt)

    if self_train:
        # Self-supervised recalibration: adopt confidently-accepted test batches
        # as extra normal reference (unlabelled, no leakage of labels).
        conf = s_test < min(thr, np.quantile(oof[y == 0], 0.75))
        if conf.sum() > 0:
            model = ProcessModel(cfg).fit(S, Tr, y, extra_normal_Tr=Trt[conf])
            s_test = model.score(St, Trt)
            print(f"[{proc}] self-training: +{int(conf.sum())} unlabelled test batches added to normal reference")

    pred = np.where(s_test >= thr, "rejected", "accepted")
    print(f"[{proc}] test predicted rejected: {(pred == 'rejected').sum()}/{len(pred)} "
          f"(calibration rate {y.mean():.2f})")
    return pd.DataFrame({"batch_id": tids, "score": s_test, "label": pred})


def main_real(args):
    cache = args.cache
    outs = []
    for proc in ["A", "B"]:
        cal, lab, tst = load_process(args.data, proc, cache)
        r = run_process(proc, cal, lab, tst, self_train=not args.no_self_train)
        r["project"] = proc
        outs.append(r)
    res = pd.concat(outs, ignore_index=True)
    res.to_csv(args.out.replace(".csv", "_scores.csv"), index=False)

    tmpl_path = os.path.join(args.data, "BRPMM_template.csv")
    tmpl = pd.read_csv(tmpl_path)
    lab_map = res.set_index("batch_id")["label"]
    missing = set(tmpl["batch_id"]) - set(lab_map.index)
    if missing:
        raise ValueError(f"{len(missing)} template batches without prediction, e.g. {list(missing)[:3]}")
    tmpl["label"] = tmpl["batch_id"].map(lab_map)
    tmpl.to_csv(args.out, index=False)
    print(f"wrote {args.out}  ({tmpl['label'].value_counts().to_dict()})")


# -------------------------------------------------------------- synthetic test
def _synth(proc, n, rej_frac, rng, offset=0):
    rows = []
    for i in range(n):
        bad = rng.random() < rej_frac
        T = int(rng.integers(2400, 2900)) if proc == "A" else 600
        t = np.linspace(0, 1, T)[:, None]
        base = np.stack([np.sin(2 * np.pi * (t[:, 0] + 0.1 * j)) * (1 + 0.1 * j) for j in range(6)], 1)
        X = base + rng.normal(0, 0.2, (T, 6)) + rng.normal(0, 0.1, (1, 6))
        if bad:
            X[:, 2] += 0.8 * t[:, 0]                       # drift fault
            if proc == "A" and rng.random() < 0.3:
                X = X[: T // 8]                              # aborted run
        X[rng.integers(0, len(X), 3)] += 50                  # spikes
        d = pd.DataFrame(X, columns=[f"s{j}" for j in range(6)])
        d.insert(0, "batch_id", f"{proc}_{offset + i:03d}")
        d["__bad"] = int(bad)
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def selftest():
    rng = np.random.default_rng(1)
    for proc, frac in [("A", 0.25), ("B", 0.12)]:
        cal = _synth(proc, 80 if proc == "A" else 40, frac, rng)
        tst = _synth(proc, 80 if proc == "A" else 40, frac, rng, offset=1000)
        lab = (cal.groupby("batch_id", sort=False)["__bad"].first()
               .map({1: "rejected", 0: "accepted"}).rename("label").reset_index())
        truth = tst.groupby("batch_id", sort=False)["__bad"].first()
        r = run_process(proc, cal.drop(columns="__bad"), lab, tst.drop(columns="__bad"))
        yt = truth.reindex(r["batch_id"]).to_numpy()
        print(f"[{proc}] synthetic TEST AUC={roc_auc_score(yt, r['score']):.3f} "
              f"balAcc={balanced_accuracy_score(yt, (r['label'] == 'rejected').astype(int)):.3f}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../input-data/Project 2 - BRPMM")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--cache", default="cache", help="parquet cache dir (slow mount)")
    ap.add_argument("--no-self-train", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    selftest() if a.selftest else main_real(a)
