"""Project 2 - BRPMM: resilient batch process monitoring (first solution).

Strategy (shared feature extraction, process-specific models):

1. Per-batch featurisation (robust to sensor outliers)
   - summary features per sensor (median, MAD, quantiles, slope, start/end level,
     roughness) + batch length (A only);
   - time-normalised trajectory per sensor (K bins, bin medians) so batches of
     different length / phase are comparable.
2. Process A uses class-weighted Extra Trees on selected batch summaries,
   mode-specific sensor deviation, relative peak timing, and maximum rolling
   MAD, plus targeted temperature, pressure, and density features. Batches
   outside learned sensor limits rank first. The cutoff maximizes accepted-class F1 on
   out-of-fold predictions.
3. Process B uses two complementary detectors
   - MONITOR (unsupervised, "what is normal"): batch-wise PCA fitted on
     ACCEPTED batches only -> Hotelling T^2 + SPE (Q), normalised by empirical
     control limits.  Needs no rejected examples, so it copes with the heavy
     class imbalance (B: 5 rejected) and flags only deviations that break the
     correlation structure, not every new-but-acceptable level shift.
   - CLASSIFIER (supervised, "what is bad"): class-weighted logistic regression
     + extra-trees on the summary features -> discrimination near the border
     between accepted and rejected batches.
   The final score blends both; the decision threshold is chosen on
   out-of-fold (repeated stratified CV) scores.
4. Process B recalibration (enabled unless --no-self-train is set): confidently
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
    "A": dict(n_bins=30, pca_var=0.90, use_length=True, drop=[], w_monitor=0.5,
              use_rolling_mad=True, use_mad_persistence=False, use_targeted_features=True),
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


def _rolling_mad(X: np.ndarray) -> np.ndarray:
    """Window MAD per sensor using overlapping 1/30-batch windows."""
    length = len(X)
    width = min(length, max(5, round(length / 30)))
    stride = max(1, width // 4)
    starts = np.r_[np.arange(0, length - width, stride), length - width]
    windows = np.lib.stride_tricks.sliding_window_view(X, width, axis=0)[starts]
    centers = np.median(windows, axis=-1, keepdims=True)
    return np.median(np.abs(windows - centers), axis=-1)


def batch_features(
    X: np.ndarray, n_bins: int, use_length: bool, *,
    use_rolling_mad=False, retain_mad_windows=False, targeted_sensors=None,
):
    """Return batch summaries and sequence features for one batch."""
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
    if use_rolling_mad:
        rolling_mad = _rolling_mad(X)
        traj.append(rolling_mad.max(axis=0))
    if targeted_sensors is not None:
        temperature, pressure, density = targeted_sensors
        temperature_bins = traj[temperature]
        dip = temperature_bins[7:23].min() - np.median(temperature_bins[:5])

        start, stop = int(T * 0.05), int(T * 0.95)
        steady_pressure = X[start:stop, pressure] if stop - start > 50 else X[:, pressure]
        window = max(11, len(steady_pressure) // 20)
        smooth = pd.Series(steady_pressure).rolling(window, center=True, min_periods=1).median()
        wander = np.subtract(*np.percentile(smooth, [97.5, 2.5]))
        traj.append(np.array([dip, wander, np.median(X[:, density])]))
    if retain_mad_windows:
        traj.append(rolling_mad.ravel())
    return np.array(summ, float), np.concatenate(traj)


def featurise(df: pd.DataFrame, cfg: dict):
    drop = [c for c in cfg["drop"] if c in df.columns]
    cols = [c for c in df.columns if c != "batch_id" and c not in drop]
    retain_mad_windows = cfg.get("use_mad_persistence", False)
    targeted_sensors = None
    if cfg.get("use_targeted_features", False):
        targeted_sensors = tuple(cols.index(name) for name in ("temperature_1", "pressure_4", "density"))
    ids, S, Tr = [], [], []
    for bid, g in df.groupby("batch_id", sort=False):
        s, tr = batch_features(
            g[cols].to_numpy(float), cfg["n_bins"], cfg["use_length"],
            use_rolling_mad=cfg.get("use_rolling_mad", False),
            retain_mad_windows=retain_mad_windows, targeted_sensors=targeted_sensors,
        )
        ids.append(bid), S.append(s), Tr.append(tr)
    if retain_mad_windows:
        # Window counts vary with batch length. Padding is ignored by the model.
        width = max(map(len, Tr))
        Tr = [np.pad(tr, (0, width - len(tr)), constant_values=np.nan) for tr in Tr]
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


class ProcessAModel:
    """Class-weighted Extra Trees with a sensor-limit ranking rule."""

    # Midpoint between the two typical operating lengths, rounded to a row.
    MODE_LENGTH = 2548
    ENVELOPE_MODE_LENGTH = 2550
    # Each sensor contributes 14 statistics. Its median is the third.
    SENSOR_MEDIANS = slice(2, -1, 14)
    # Keep the 5th, 50th and 95th percentiles, MAD, slope and roughness.
    SUMMARY_STATS = (0, 2, 4, 5, 10, 13)

    def _long_mode(self, summaries):
        return summaries[:, -1] >= np.log(self.MODE_LENGTH)

    def _with_deviation_score(self, summaries):
        long_mode = self._long_mode(summaries)
        scores = np.empty(len(summaries))
        for mode, (center, lower, upper) in enumerate(self.references):
            mask = long_mode == mode
            delta = summaries[mask, self.SENSOR_MEDIANS] - center
            scores[mask] = np.maximum(delta / upper, -delta / lower).max(axis=1)
        sensor_stats = summaries[:, :-1].reshape(len(summaries), -1, 14)
        selected = sensor_stats[:, :, self.SUMMARY_STATS].reshape(len(summaries), -1)
        return np.column_stack((selected, summaries[:, -1], scores))

    def _classifier_inputs(self, summaries, trajectories):
        n_sensors = summaries[:, self.SENSOR_MEDIANS].shape[1]
        split = n_sensors * self.n_bins
        windows = trajectories[:, :split].reshape(len(summaries), n_sensors, self.n_bins)
        peak_timing = windows.argmax(axis=2) / (self.n_bins - 1)
        end_maxima = split + n_sensors
        rolling_maxima = trajectories[:, split:end_maxima]
        inputs = [self._with_deviation_score(summaries), peak_timing, rolling_maxima]
        mode = self._envelope_long_mode(summaries).astype(int)
        if self.use_targeted_features:
            targeted = trajectories[:, end_maxima:end_maxima + 3].copy()
            targeted[:, 2] -= self.density_centers[mode]
            inputs.append(targeted)
            end_maxima += 3
        if self.use_mad_persistence:
            rolling = trajectories[:, end_maxima:].reshape(len(summaries), -1, n_sensors)
            above = rolling > self.mad_limits[mode, None, :]
            window_count = np.isfinite(rolling).sum(axis=1)
            inputs.append(above.sum(axis=1) / window_count)
        return np.column_stack(inputs)

    def _envelope_long_mode(self, summaries):
        return summaries[:, -1] >= np.log(self.ENVELOPE_MODE_LENGTH)

    def _envelope_shifts(self, summaries):
        long_mode = self._envelope_long_mode(summaries)
        centers = self.envelope_centers[long_mode.astype(int)]
        return np.abs(summaries[:, self.SENSOR_MEDIANS] - centers)

    def __init__(self, cfg):
        self.n_bins = cfg["n_bins"]
        self.use_mad_persistence = cfg.get("use_mad_persistence", False)
        self.use_targeted_features = cfg.get("use_targeted_features", False)
        self.classifier = ExtraTreesClassifier(
            n_estimators=160, min_samples_leaf=1, class_weight="balanced",
            random_state=SEED, n_jobs=2,
        )

    def fit(self, S, Tr, y):
        accepted = S[y == 0]
        long_mode = self._long_mode(accepted)
        levels = accepted[:, self.SENSOR_MEDIANS]
        self.references = []
        for mode in (False, True):
            reference = levels[long_mode == mode]
            if not len(reference):
                reference = levels
            center = np.median(reference, axis=0)
            residual = reference - center
            floor = np.maximum(1.4826 * np.median(np.abs(residual), axis=0), 1e-6)
            lower = np.maximum(-np.quantile(residual, 0.02, axis=0), floor)
            upper = np.maximum(np.quantile(residual, 0.98, axis=0), floor)
            self.references.append((center, lower, upper))
        # The sensor envelope uses the validated 2,550-row split and a
        # symmetric limit shared by both modes for each sensor.
        envelope_long = self._envelope_long_mode(accepted)
        n_sensors = levels.shape[1]
        maxima_start = n_sensors * self.n_bins
        if self.use_targeted_features:
            density = Tr[y == 0, maxima_start + n_sensors + 2]
            self.density_centers = np.array([
                np.median(density[envelope_long == mode]) if (envelope_long == mode).any() else np.median(density)
                for mode in (False, True)
            ])
        if self.use_mad_persistence:
            normal_maxima = Tr[y == 0, maxima_start:maxima_start + n_sensors]
            limits = []
            for mode in (False, True):
                reference = normal_maxima[envelope_long == mode]
                limits.append(np.quantile(reference if len(reference) else normal_maxima, 0.98, axis=0))
            self.mad_limits = np.stack(limits)
        centers = []
        for mode in (False, True):
            reference = levels[envelope_long == mode]
            centers.append(np.median(reference if len(reference) else levels, axis=0))
        self.envelope_centers = np.stack(centers)
        shifts = self._envelope_shifts(accepted)
        self.envelope_limits = np.maximum(
            1.5 * shifts.max(axis=0), 8 * 1.4826 * np.median(shifts, axis=0) + 1e-12,
        )
        self.classifier.fit(self._classifier_inputs(S, Tr), y)
        return self

    def score(self, S, Tr):
        probability = self.classifier.predict_proba(self._classifier_inputs(S, Tr))[:, 1]
        outside_limits = (self._envelope_shifts(S) > self.envelope_limits).any(axis=1)
        return probability + outside_limits


# ------------------------------------------------------------------ evaluation
def process_a_threshold(y, scores):
    """Choose the rejection cutoff that maximizes accepted-class F1."""
    cuts = np.unique(np.r_[scores, np.nextafter(scores.max(), np.inf)])
    predicted_accepted = scores[:, None] < cuts
    accepted = y == 0
    true_accepted = predicted_accepted[accepted].sum(axis=0)
    f1 = 2 * true_accepted / (predicted_accepted.sum(axis=0) + accepted.sum())
    best = np.flatnonzero(f1 == f1.max())
    return float(cuts[best[len(best) // 2]])


def best_threshold(y, s):
    cand = np.unique(np.quantile(s, np.linspace(0.02, 0.98, 97)))
    bal = np.array([balanced_accuracy_score(y, s >= c) for c in cand])
    ok = cand[bal >= bal.max() - 1e-9]
    return float(np.median(ok))  # middle of the best plateau = more stable


def oof_scores(S, Tr, y, cfg, n_splits=5, n_repeats=3, model_class=ProcessModel):
    n = len(y)
    acc, cnt = np.zeros(n), np.zeros(n)
    n_splits = max(2, min(n_splits, int(y.sum())))
    rs = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
    for tr, te in rs.split(S, y):
        m = model_class(cfg).fit(S[tr], Tr[tr], y[tr])
        acc[te] += m.score(S[te], Tr[te])
        cnt[te] += 1
    return acc / cnt


def report(name, y, s, thr):
    pred = (s >= thr).astype(int)
    print(f"[{name}] OOF  AUC={roc_auc_score(y, s):.3f}  balAcc={balanced_accuracy_score(y, pred):.3f}  "
          f"MCC={matthews_corrcoef(y, pred):.3f}  F1(acc)={f1_score(y, pred, pos_label=0):.3f}  "
          f"F1(rej)={f1_score(y, pred):.3f}  thr={thr:.3f}")


# ----------------------------------------------------------------------- driver
def run_process(proc, cal, lab, tst, self_train=True):
    cfg = CFG[proc]
    ids, S, Tr = featurise(cal, cfg)
    y = labels_to_y(lab, ids)
    tids, St, Trt = featurise(tst, cfg)
    print(f"[{proc}] cal={len(ids)} (rejected={y.sum()}), test={len(tids)}, "
          f"features: summary={S.shape[1]}, sequence={Tr.shape[1]}")

    model_class = ProcessAModel if proc == "A" else ProcessModel
    choose_threshold = process_a_threshold if proc == "A" else best_threshold
    oof = oof_scores(S, Tr, y, cfg, model_class=model_class)
    thr = choose_threshold(y, oof)
    report(proc, y, oof, thr)

    model = model_class(cfg).fit(S, Tr, y)
    s_test = model.score(St, Trt)

    if self_train and proc == "B":
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
        columns = ["s0", "density", "temperature_1", "s3", "s4", "pressure_4"] if proc == "A" else [
            f"s{j}" for j in range(6)
        ]
        d = pd.DataFrame(X, columns=columns)
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
