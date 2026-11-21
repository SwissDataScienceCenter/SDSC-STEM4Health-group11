"""Process A: sensor-wise, mode-aware batch monitoring.

Why (from looking at the raw data, see notes in the PR / slides):
  * Process A is a steady-state process: every sensor holds a flat level for
    the whole batch. Trajectory shape carries little information.
  * Faults are mostly ONE sensor at a shifted level / with extra noise or
    oscillation, a different sensor each time -> the variables are largely
    independent, so a single global PCA dilutes them.
  * Shifts come in sizes: small shifts are still 'accepted', large ones are
    'rejected' -> we need per-sensor acceptable ranges.
  * Two operating modes (batch length ~2370 vs ~2725 rows) with different
    normal levels (temperature_2, pressure_2/3/4) -> compare each batch only
    with good batches of the same mode.

Pipeline:
  1. per-sensor, per-batch statistics (level, noise, wander, oscillation,
     spikes, start/end transients) on the steady part of the batch
  2. robust z-score of every statistic vs. GOOD training batches
  3. learned per-sensor safe ranges (Envelope): any set-point shift larger
     than ever seen in an accepted batch -> rejected
  4. ExtraTrees classifier on the deviations (learns which deviation sizes
     are still acceptable), trained only on accepted vs the SUBTLE rejected
     batches the safe ranges don't flag (SubtleModel). An unsupervised worst-sensor monitor is kept
     for interpretation only.

CV result (5 seeds x 5-fold x 2 repeats), grading metric F1(accepted):
  global-PCA model (brpmm.py) 0.891 | ExtraTrees 0.911 | + Envelope 0.931
  FINAL SubtleModel (trees trained only on the cases the Envelope misses):
  AUC 0.927 -> 0.956, rejected caught 28 -> 35 / 40 (see SubtleModel)
Tried without gain: block/per-sensor PCA, shakiness, over/undersampling,
mode-aware scaling, event/oscillation-amplitude/step features, smaller
feature sets, ET hyper-parameters (all within +-0.005).

Usage:
    python src/sensorwise.py            # cross-validated comparison
    python src/sensorwise.py --predict  # also write test predictions for A
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

sys.path.insert(0, os.path.dirname(__file__))
from brpmm import best_threshold  # noqa: E402

SEED = 0
ROOT = "../input-data/Project 2 - BRPMM/process A"
CACHE = "../cache"
MODE_SPLIT = 2550  # rows: short mode ~2370, long mode ~2725


# ------------------------------------------------------------- featurisation
def _rolling_median(x, w):
    return pd.Series(x).rolling(w, center=True, min_periods=1).median().to_numpy()


def _autocorr_peak(r, lags=range(5, 120, 5)):
    r = r - r.mean()
    v = (r * r).sum()
    if v <= 0 or len(r) < 150:
        return 0.0
    return max(float((r[:-k] * r[k:]).sum() / v) for k in lags)


def batch_stats(X, cols):
    T = len(X)
    a, b = int(T * 0.05), int(T * 0.95)
    out = {"len": T, "log_len": np.log(T)}
    for j, c in enumerate(cols):
        x = X[:, j]
        xs = x[a:b] if b - a > 50 else x
        lvl = np.median(xs)
        hf = xs - _rolling_median(xs, 11)                 # fast noise
        slow = _rolling_median(xs, max(11, len(xs) // 20))  # slow drift
        hf_sd = 1.4826 * np.median(np.abs(hf)) + 1e-9
        out[f"{c}:level"] = lvl
        out[f"{c}:noise"] = hf_sd
        out[f"{c}:spread"] = np.subtract(*np.percentile(xs, [97.5, 2.5]))
        out[f"{c}:wander"] = np.subtract(*np.percentile(slow, [97.5, 2.5]))
        out[f"{c}:osc"] = _autocorr_peak(xs - _rolling_median(xs, 301))
        out[f"{c}:spikes"] = np.mean(np.abs(hf) > 6 * hf_sd)
        out[f"{c}:tail"] = np.percentile(np.abs(hf), 99.5) / hf_sd
        n = max(5, T // 20)
        out[f"{c}:start"] = np.median(x[:n]) - lvl
        out[f"{c}:end"] = np.median(x[-n:]) - lvl
    return out


def load_feats(kind):
    """kind = 'cal' or 'test'. Cached per-batch statistics table."""
    cp = os.path.join(CACHE, f"A_{kind}_stats.parquet")
    if os.path.exists(cp):
        return pd.read_parquet(cp)
    sub = "calibration data/calibration_data.csv" if kind == "cal" else "test data/test_data.csv"
    pq = os.path.join(CACHE, f"A_{kind}.parquet")
    df = pd.read_parquet(pq) if os.path.exists(pq) else pd.read_csv(os.path.join(ROOT, sub))
    cols = [c for c in df.columns if c != "batch_id"]
    F = pd.DataFrame({bid: batch_stats(g[cols].to_numpy(float), cols)
                      for bid, g in df.groupby("batch_id", sort=False)}).T.astype(float)
    os.makedirs(CACHE, exist_ok=True)
    F.to_parquet(cp)
    return F


# ---------------------------------------------------------------------- model
class ModeScaler:
    """Robust z-scores vs. GOOD training batches (optionally per operating mode).

    mode_aware=False pools both modes: tested over 5 CV seeds it was slightly
    better (AUC 0.898 vs 0.889) because the classifier also gets a 'short'
    flag and learns the mode offsets itself.
    """

    def __init__(self, mode_aware=False):
        self.mode_aware = mode_aware

    def _mode(self, F):
        return (F["len"] < MODE_SPLIT).to_numpy() if self.mode_aware else np.zeros(len(F), bool)

    def fit(self, F, y):
        self.cols = [c for c in F.columns if c not in ("len", "log_len")]
        mode = self._mode(F)
        self.ref = {}
        for m in (False, True):
            G = F.loc[(y == 0) & (mode == m), self.cols]
            if len(G) < 8:  # too few good batches in this mode -> pool both
                G = F.loc[y == 0, self.cols]
            med = G.median()
            mad = (G - med).abs().median() * 1.4826
            sd = G.std()
            # floor the scale: MAD can be ~0 for rarely-moving stats
            floor = 0.1 * sd.where(sd > 0, 1.0) + 1e-9
            self.ref[m] = (med, np.maximum(mad, floor))
        return self

    def transform(self, F):
        mode = self._mode(F)
        Z = pd.DataFrame(index=F.index, columns=self.cols, dtype=float)
        for m in (False, True):
            med, sc = self.ref[m]
            idx = mode == m
            Z.loc[idx] = ((F.loc[idx, self.cols] - med) / sc).to_numpy()
        return Z


LEVEL_SENSORS = ["flow_rate_1", "density", "flow_rate_2", "pressure_1", "temperature_1",
                 "temperature_2", "temperature_3", "temperature_4", "pressure_2",
                 "pressure_3", "pressure_4"]


class Envelope:
    """Learned per-sensor safe range (SPC-style, easy to explain and recalibrate).

    Process A faults are set-point shifts of fixed sizes (e.g. temperature_4:
    accepted +-1, rejected +-3). A batch is flagged if any sensor's level is
    shifted further than `margin` x the largest shift seen in an ACCEPTED
    training batch (and > zmin robust sd, so normal noise never triggers).
    Shift = level minus the good-batch median of the same operating mode.
    CV: flags ~16/40 bad batches at ~3/121 false alarms; ET + envelope raises
    F1(accepted) 0.911 -> 0.931 (5 seeds). Recalibration = refit on new data.
    """

    def __init__(self, margin=1.5, zmin=8):
        self.margin, self.zmin = margin, zmin

    def _shift(self, F):
        short = (F["len"] < MODE_SPLIT).to_numpy()
        return pd.DataFrame({c: F[f"{c}:level"].to_numpy()
                             - np.where(short, self.nom[True][c], self.nom[False][c])
                             for c in LEVEL_SENSORS}, index=F.index)

    def fit(self, F, y):
        short = (F["len"] < MODE_SPLIT).to_numpy()
        self.nom = {m: {c: F.loc[(y == 0) & (short == m), f"{c}:level"].median()
                        for c in LEVEL_SENSORS} for m in (False, True)}
        G = self._shift(F).abs()[y == 0]
        self.limit = np.maximum(self.margin * G.max(), self.zmin * 1.4826 * G.median() + 1e-12)
        return self

    def ratio(self, F):
        """>1 = outside the safe range on that sensor (useful for explanations)."""
        return self._shift(F).abs() / self.limit

    def flag(self, F):
        return (self.ratio(F) > 1).any(axis=1).to_numpy()


def _signed_log(z):
    return np.sign(z) * np.log1p(np.abs(z))


class SensorwiseModel:
    """Final score = ExtraTrees on per-sensor deviations.

    The unsupervised 'worst sensor' monitor is kept for interpretation
    (parts()[1]) but not blended in: accepted batches can also carry large
    (acceptable) shifts, so the monitor alone only reaches AUC ~0.68 and
    blending lowered CV performance.
    """

    def __init__(self, mode_aware=False):
        self.mode_aware = mode_aware

    def _design(self, F):
        Z = self.sc.transform(F)
        A = np.abs(Z)
        types = sorted({c.split(":")[1] for c in Z.columns})
        # worst deviation per statistic type (sensor-agnostic -> generalises
        # to a fault on a sensor never seen faulty in training)
        worst = pd.DataFrame({f"worst_{t}": A[[c for c in A if c.endswith(":" + t)]].max(axis=1)
                              for t in types})
        X = pd.concat([_signed_log(Z), np.log1p(worst),
                       F[["log_len"]], (F["len"] < MODE_SPLIT).astype(float).rename("short")], axis=1)
        return X.to_numpy(float), np.log1p(worst.to_numpy(float))

    def fit(self, F, y):
        self.sc = ModeScaler(self.mode_aware).fit(F, y)
        X, W = self._design(F)
        self.et = ExtraTreesClassifier(n_estimators=600, min_samples_leaf=2, max_features=0.3,
                                       class_weight="balanced_subsample", random_state=SEED,
                                       n_jobs=-1).fit(X, y)
        # monitor reference: worst-deviation distribution of good batches
        self.mon_ref = np.sort(W[y == 0].max(1))
        self.env = Envelope().fit(F, y)
        return self

    def parts(self, F):
        X, W = self._design(F)
        p_sup = self.et.predict_proba(X)[:, 1]
        # monitor: percentile of worst deviation among good training batches
        mon = np.searchsorted(self.mon_ref, W.max(1)) / (len(self.mon_ref) + 1)
        return p_sup, mon

    def score(self, F):
        """Envelope-flagged batches rank first (score > 1), then by ExtraTrees."""
        p = self.parts(F)[0]
        return np.where(self.env.flag(F), 1.0 + p, p)


class SubtleModel(SensorwiseModel):
    """Final model: safe ranges for the obvious faults, trees for the subtle ones.

    The Envelope already catches batches with a big out-of-range shift. If the
    trees are also trained on those, they spend their effort on "big shift =
    bad" and get sloppy at the fine boundary. So the trees learn only from
    accepted vs SUBTLE rejected batches (those the Envelope does not flag),
    on robust z-scores computed per operating mode.

    Subtle rejected batches show e.g. a temperature_1 dip mid-batch and larger
    pressure_4 swings; individually small, but consistent in combination.
    CV (same pipeline as --predict, 5 fresh seeds): F1(accepted) 0.911 -> 0.938,
    AUC 0.927 -> 0.956, rejected caught 28 -> 35 / 40 at 5 -> 6 / 121 false alarms.
    """

    def _z(self, F):
        X = F.to_numpy(float)
        short = (F["len"] < MODE_SPLIT).to_numpy()
        Z = np.zeros(X.shape)
        for m in (False, True):
            med, sc = self.ref[m]
            Z[short == m] = (X[short == m] - med) / sc
        return np.nan_to_num(np.clip(Z, -50, 50))

    def fit(self, F, y):
        self.env = Envelope().fit(F, y)
        X = F.to_numpy(float)
        short = (F["len"] < MODE_SPLIT).to_numpy()
        self.ref = {}
        for m in (False, True):
            G = X[(y == 0) & (short == m)]
            med = np.nanmedian(G, 0)
            self.ref[m] = (med, np.nanmedian(np.abs(G - med), 0) * 1.4826 + 1e-9)
        subtle = (y == 0) | ~self.env.flag(F)
        self.et = ExtraTreesClassifier(n_estimators=800, min_samples_leaf=2, max_features="sqrt",
                                       class_weight="balanced", random_state=SEED,
                                       n_jobs=-1).fit(self._z(F)[subtle], y[subtle])
        return self

    def score(self, F):
        p = self.et.predict_proba(self._z(F))[:, 1]
        return np.where(self.env.flag(F), 1.0 + p, p)


# ----------------------------------------------------------------- evaluation
def labels(F):
    lab = pd.read_csv(os.path.join(ROOT, "calibration data/calibration_labels.csv"))
    return (lab.set_index("batch_id")["label"].reindex(F.index) == "rejected").astype(int).to_numpy()


def cv_scores(F, y, make, n_repeats=3):
    acc, cnt = np.zeros(len(y)), np.zeros(len(y))
    rs = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=SEED)
    for tr, te in rs.split(F, y):
        acc[te] += make(F.iloc[tr], y[tr])(F.iloc[te])
        cnt[te] += 1
    return acc / cnt


def show(name, y, s):
    pred = s >= best_threshold(y, s)
    print(f"  {name:<40} AUC={roc_auc_score(y, s):.3f}  balAcc={balanced_accuracy_score(y, pred):.3f}  "
          f"bad caught {int((pred & (y == 1)).sum()):>2}/{int(y.sum())}  "
          f"good flagged {int((pred & (y == 0)).sum()):>2}/{int((y == 0).sum())}")


def baseline_oof(ids):
    """Teammate's model on identical splits (needs ../cache/feats_A.npz)."""
    from brpmm import CFG, ProcessModel
    z = np.load(os.path.join(CACHE, "feats_A.npz"), allow_pickle=True)
    assert list(z["ids"]) == list(ids)
    S, Tr = z["S"], z["Tr"]
    return S, Tr, lambda tr, y: ProcessModel(CFG["A"]).fit(S[tr], Tr[tr], y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predict", action="store_true")
    a = ap.parse_args()

    F = load_feats("cal")
    y = labels(F)
    print(f"Process A: {len(y)} batches, {int(y.sum())} rejected\n")

    mon = cv_scores(F, y, lambda Ft, yt: (lambda m: lambda Fe: m.parts(Fe)[1])(SensorwiseModel().fit(Ft, yt)))
    sup = cv_scores(F, y, lambda Ft, yt: SensorwiseModel().fit(Ft, yt).score)
    sub = cv_scores(F, y, lambda Ft, yt: SubtleModel().fit(Ft, yt).score)
    show("monitor only (no bad labels used)", y, mon)
    show("sensor-wise model (previous)", y, sup)
    show("FINAL: subtle-trained model", y, sub)

    if os.path.exists(os.path.join(CACHE, "feats_A.npz")):
        S, Tr, mk = baseline_oof(F.index)
        acc, cnt = np.zeros(len(y)), np.zeros(len(y))
        for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED).split(F, y):
            acc[te] += mk(tr, y[tr]).score(S[te], Tr[te])
            cnt[te] += 1
        show("teammate's model (for comparison)", y, acc / cnt)

    if a.predict:
        Ft = load_feats("test")
        m = SubtleModel().fit(F, y)
        # Grading metric = F1 with 'accepted' as the positive class.
        # Pick the fraction of batches to flag as rejected that maximises it
        # out-of-fold, then flag that same fraction of the test batches
        # (a fraction transfers more robustly than a raw score threshold).
        from sklearn.metrics import f1_score
        oof = cv_scores(F, y, lambda Ftr, ytr: SubtleModel().fit(Ftr, ytr).score, n_repeats=10)
        qs = np.linspace(0.05, 0.5, 46)
        f1s = [f1_score(1 - y, (oof < np.quantile(oof, 1 - q)).astype(int)) for q in qs]
        q = qs[int(np.argmax(f1s))]
        print(f"\nflag top {q:.0%} as rejected (OOF F1 accepted = {max(f1s):.3f})")
        s = m.score(Ft)
        thr = np.quantile(s, 1 - q)
        out = pd.DataFrame({"batch_id": Ft.index, "project": "A", "score": s,
                            "label": np.where(s >= thr, "rejected", "accepted")})
        out.to_csv("predictions_A_sensorwise.csv", index=False)
        print(f"\nwrote predictions_A_sensorwise.csv: {(out.label == 'rejected').sum()}/{len(out)} rejected")


if __name__ == "__main__":
    main()
