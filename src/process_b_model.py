"""Process B — modelling strategy (used by src/process_b/process_b_MODELING.ipynb).

Two complementary layers plus an update loop:

1. Supervised classifier  (LogisticRegression on per-batch mean/range of 3 sensors)
   -> catches the *known* failure signature (aeration dip at 10–25 % of the batch).
2. Unsupervised MSPC monitor  (batch-wise unfolded PCA with Hotelling T² and SPE, fitted on
   ACCEPTED batches only, on set-point-centred trajectories of ALL 20 sensors)
   -> catches *unseen* deviations on any sensor and needs no rejected examples.
3. Self-training recalibration
   -> new batches that both layers confidently agree on are pseudo-labelled and folded back
      into the reference set, so the models (and control limits) update without new lab labels.

Final decision: rejected if the classifier says so OR the MSPC raises an alarm.

Note: src/process_b.py is a separate config file for pipeline.py/benchmark.py; this module is
self-contained.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")
SEED = 42

# --------------------------------------------------------------------------- data
def _find_data() -> Path:
    """Locate 'input-data/Project 2 - BRPMM': $BRPMM_DATA, then any parent folder, then Renku default."""
    if os.environ.get("BRPMM_DATA"):
        return Path(os.environ["BRPMM_DATA"])
    for parent in [Path.cwd(), *Path.cwd().parents, *Path(__file__).resolve().parents]:
        cand = parent / "input-data" / "Project 2 - BRPMM"
        if cand.is_dir():
            return cand
    return Path("/home/renku/work/input-data/Project 2 - BRPMM")


DEFAULT_DATA = _find_data()


def load_process_b(root: str | Path | None = None):
    base = Path(root or DEFAULT_DATA) / "process B"
    cal = pd.read_csv(base / "calibration data" / "calibration_data.csv")
    lab = pd.read_csv(base / "calibration data" / "calibration_labels.csv")
    tst = pd.read_csv(base / "test data" / "test_data.csv")
    return cal, lab, tst


# ------------------------------------------------------------- classifier features
KEY_SENSORS = ["aeration_rate", "dissolved_oxygen", "biomass_concentration",
               "fermentation_volume", "feed_temperature", "substrate_concentration"]
TOP3_SENSORS = ["aeration_rate", "fermentation_volume", "feed_temperature"]


def compute_batch_stats(df: pd.DataFrame, sensors) -> pd.DataFrame:
    """Per batch and sensor: mean and range (max - min)."""
    g = df.groupby("batch_id", sort=True)[list(sensors)]
    mean, rng = g.mean(), g.max() - g.min()
    out = pd.DataFrame(index=mean.index)
    for s in sensors:
        out[f"{s}_mean"] = mean[s]
        out[f"{s}_range"] = rng[s]
    return out.reset_index()


def prepare_data(df_calib, df_labels, sensors):
    feat = compute_batch_stats(df_calib, sensors).merge(df_labels, on="batch_id")
    cols = [c for c in feat.columns if c not in ("batch_id", "label")]
    le = LabelEncoder().fit(["accepted", "rejected"])  # accepted=0, rejected=1
    return feat[cols].fillna(0).values, le.transform(feat["label"]), cols, le


def make_pipeline(clf, use_smote: bool = False, k_neighbors: int = 2):
    """StandardScaler (+ SMOTE, applied inside CV folds only) + classifier."""
    if use_smote:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline
        return Pipeline([("scale", StandardScaler()),
                         ("smote", SMOTE(k_neighbors=k_neighbors, random_state=SEED)),
                         ("clf", clf)])
    from sklearn.pipeline import Pipeline
    return Pipeline([("scale", StandardScaler()), ("clf", clf)])


def make_lr():
    return LogisticRegression(max_iter=3000, random_state=SEED)


_WARNED = False


def get_models():
    models = {"LR": make_lr,
              "RF": lambda: RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                   random_state=SEED)}
    try:
        from xgboost import XGBClassifier
        models["XGB"] = lambda: XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1,
                                              scale_pos_weight=7, random_state=SEED,
                                              eval_metric="logloss", verbosity=0)
    except Exception:  # e.g. libomp missing on macOS
        from sklearn.ensemble import HistGradientBoostingClassifier
        global _WARNED
        if not _WARNED:
            print("[info] xgboost unavailable (e.g. libomp missing) -> sklearn HistGradientBoosting as 'GB'")
            _WARNED = True
        models["GB"] = lambda: HistGradientBoostingClassifier(max_depth=3, min_samples_leaf=2,
                                                              class_weight="balanced", random_state=SEED)
    return models


METRICS = {"accuracy": accuracy_score,
           "precision": lambda a, b: precision_score(a, b, zero_division=0),
           "recall": lambda a, b: recall_score(a, b, zero_division=0),
           "f1": lambda a, b: f1_score(a, b, zero_division=0)}


def cv_scores(X, y, model_fn, use_smote=False, n_splits=5, seed=SEED):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out = {m: [] for m in METRICS}
    for tr, te in skf.split(X, y):
        pipe = make_pipeline(model_fn(), use_smote).fit(X[tr], y[tr])
        pred = pipe.predict(X[te])
        for m, f in METRICS.items():
            out[m].append(f(y[te], pred))
    return {m: (float(np.mean(v)), float(np.std(v))) for m, v in out.items()}


def run_benchmark(df_calib, df_labels, verbose=True):
    """All models x {12, 6 features} x {no SMOTE, SMOTE}, 5-fold stratified CV."""
    results = []
    for fs_name, sensors in [("12 features", KEY_SENSORS), ("6 features (top 3)", TOP3_SENSORS)]:
        X, y, _, _ = prepare_data(df_calib, df_labels, sensors)
        for mname, fn in get_models().items():
            for smote in (False, True):
                cfg = f"{mname} | {fs_name} | {'SMOTE' if smote else 'no SMOTE'}"
                r = {"config": cfg, **cv_scores(X, y, fn, smote)}
                results.append(r)
    if verbose:
        print(f"{'Config':<40s}" + "".join(f"{m:>16s}" for m in METRICS))
        print("-" * 104)
        for r in results:
            print(f"{r['config']:<40s}" + "".join(f"{r[m][0]:>9.3f} ±{r[m][1]:.2f}" for m in METRICS))
    return results


def analyze_coefficients(df_calib, df_labels, sensors=KEY_SENSORS, verbose=True):
    X, y, cols, _ = prepare_data(df_calib, df_labels, sensors)
    pipe = make_pipeline(make_lr()).fit(X, y)
    coef = pipe.named_steps["clf"].coef_[0]
    df = pd.DataFrame({"feature": cols, "coefficient": coef, "abs_coef": np.abs(coef)})
    df = df.sort_values("abs_coef", ascending=False).reset_index(drop=True)
    if verbose:
        print(df.round(4).to_string(index=False))
    return df


def fit_classifier(df_calib, df_labels, sensors=TOP3_SENSORS):
    X, y, cols, _ = prepare_data(df_calib, df_labels, sensors)
    return make_pipeline(make_lr()).fit(X, y), cols


def classifier_proba(pipe, cols, df, sensors=TOP3_SENSORS) -> pd.Series:
    feat = compute_batch_stats(df, sensors)
    return pd.Series(pipe.predict_proba(feat[cols].fillna(0).values)[:, 1], index=feat["batch_id"])


# ------------------------------------------------------------------- MSPC monitor
def unfold(df: pd.DataFrame, sensors, n_bins: int = 50):
    """Batch-wise unfolding: each batch -> n_bins time-bin means per sensor  (N x n_bins x S)."""
    ids, arr = [], []
    for b, g in df.groupby("batch_id", sort=True):
        X = g[list(sensors)].to_numpy(float)
        arr.append(np.stack([seg.mean(0) for seg in np.array_split(X, n_bins)]))
        ids.append(b)
    return ids, np.stack(arr)


class BatchMSPC:
    """Multivariate statistical batch-process monitor (PCA + Hotelling T² + SPE).

    * fitted on accepted batches only (no rejected examples needed);
    * each batch's trajectories are centred on its own median per sensor, so a new but benign
      set-point (e.g. aeration 4.8 vs 8.1 — both seen in accepted batches) is not an alarm;
      the monitor reacts to *shape* changes: dips, steps, drifts, oscillations;
    * all sensors are used, so faults on variables the classifier ignores are still caught;
    * control limits come from leave-one-out scores of the reference batches (honest new-batch
      behaviour): SPE via Box's g·chi² approximation, T² via the F distribution.
    """

    def __init__(self, n_bins=50, var=0.90, alpha=0.99, center=True, sd_floor=0.1):
        self.n_bins, self.var, self.alpha, self.center = n_bins, var, alpha, center
        self.sd_floor = sd_floor

    # -- helpers
    def _features(self, A):
        if self.center:
            A = A - np.median(A, axis=1, keepdims=True)
        return A.reshape(len(A), -1)

    @staticmethod
    def _fit_core(X, var, level_scale):
        mu, sd = X.mean(0), X.std(0)
        sd = np.maximum(sd, level_scale)  # floor: constant-in-reference columns stay finite
        Z = (X - mu) / sd
        n_max = max(1, min(len(X) - 2, Z.shape[1]))
        pca = PCA(n_components=var if len(X) > 3 else 1, svd_solver="full").fit(Z)
        if pca.n_components_ > n_max:
            pca = PCA(n_components=n_max, svd_solver="full").fit(Z)
        return mu, sd, pca

    @staticmethod
    def _score_core(core, X):
        mu, sd, pca = core
        Z = (X - mu) / sd
        T = pca.transform(Z)
        R = Z - pca.inverse_transform(T)
        return (T ** 2 / pca.explained_variance_).sum(1), (R ** 2).sum(1), R

    def _floor(self, A):
        """Per-sensor lower bound on column std: sd_floor x the sensor's overall spread in the
        (centred) reference trajectories, or 0.1 % of its level for sensors that are constant.
        Stops columns where all reference batches happen to agree (e.g. the exact switching time
        of the on/off feed_rate) from turning tiny benign timing shifts into huge alarms."""
        F = self._features(A).reshape(A.shape)
        spread = F.reshape(-1, A.shape[2]).std(0)
        lvl = np.median(np.abs(A), axis=(0, 1))
        return np.tile(np.maximum.reduce([self.sd_floor * spread, 1e-3 * lvl,
                                          np.full_like(lvl, 1e-6)]), self.n_bins)

    # -- API
    def fit(self, df_ref: pd.DataFrame, sensors=None):
        self.sensors = list(sensors or [c for c in df_ref.columns if c != "batch_id"])
        self.ref_ids, A = unfold(df_ref, self.sensors, self.n_bins)
        X = self._features(A)
        self.level_scale = self._floor(A)
        self.core = self._fit_core(X, self.var, self.level_scale)
        self.n_pc = self.core[2].n_components_
        # leave-one-out reference scores -> control limits
        loo_t2, loo_spe = [], []
        for i in range(len(X)):
            core_i = self._fit_core(np.delete(X, i, 0), self.var, self.level_scale)
            t2, spe, _ = self._score_core(core_i, X[[i]])
            loo_t2.append(t2[0]); loo_spe.append(spe[0])
        self.loo_t2, self.loo_spe = np.array(loo_t2), np.array(loo_spe)
        m, v = self.loo_spe.mean(), self.loo_spe.var()
        g, h = v / (2 * m), 2 * m ** 2 / v
        self.spe_lim = float(g * sps.chi2.ppf(self.alpha, h))
        n, a = len(X), self.n_pc
        self.t2_lim = float(a * (n ** 2 - 1) / (n * (n - a)) * sps.f.ppf(self.alpha, a, n - a)) \
            if n - a > 0 else float(np.percentile(self.loo_t2, 100 * self.alpha))
        return self

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        ids, A = unfold(df, self.sensors, self.n_bins)
        t2, spe, _ = self._score_core(self.core, self._features(A))
        out = pd.DataFrame({"batch_id": ids, "T2": t2, "SPE": spe})
        out["T2_ratio"], out["SPE_ratio"] = out["T2"] / self.t2_lim, out["SPE"] / self.spe_lim
        out["alarm"] = (out["T2_ratio"] > 1) | (out["SPE_ratio"] > 1)
        return out

    def contributions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-sensor share of SPE (rows = batches, columns = sensors)."""
        ids, A = unfold(df, self.sensors, self.n_bins)
        _, spe, R = self._score_core(self.core, self._features(A))
        C = (R ** 2).reshape(len(ids), self.n_bins, -1).sum(1)
        return pd.DataFrame(C / C.sum(1, keepdims=True), index=ids, columns=self.sensors)

    def contribution_over_time(self, df_batch: pd.DataFrame) -> pd.DataFrame:
        _, A = unfold(df_batch, self.sensors, self.n_bins)
        _, _, R = self._score_core(self.core, self._features(A))
        return pd.DataFrame((R ** 2).reshape(self.n_bins, -1), columns=self.sensors)


# ----------------------------------------------------------- combined strategy
def predict_strategy(df_calib, df_labels, df_new, p_thresh=0.5, verbose=True):
    """Classifier OR MSPC alarm -> rejected.  Returns per-batch table with both layers."""
    pipe, cols = fit_classifier(df_calib, df_labels)
    acc_ids = df_labels.loc[df_labels.label == "accepted", "batch_id"]
    mspc = BatchMSPC().fit(df_calib[df_calib.batch_id.isin(acc_ids)])
    tab = mspc.score(df_new)
    tab["p_rejected"] = classifier_proba(pipe, cols, df_new).loc[tab.batch_id].values
    tab["clf_reject"] = tab["p_rejected"] > p_thresh
    tab["label"] = np.where(tab["clf_reject"] | tab["alarm"], "rejected", "accepted")
    tab["reason"] = np.select(
        [tab.clf_reject & tab.alarm, tab.clf_reject, tab.alarm],
        ["known fault (both layers)", "classifier only", "novel deviation (MSPC only)"], "in control")
    if verbose:
        print(f"MSPC: {mspc.n_pc} PCs, T²-lim={mspc.t2_lim:.1f}, SPE-lim={mspc.spe_lim:.1f}")
        print("Decision counts:", tab["reason"].value_counts().to_dict())
    return tab, pipe, cols, mspc


def predict_process_b(root=None, out_path=None):
    """Final Process B submission rows (batch_id, project, label)."""
    cal, lab, tst = load_process_b(root)
    tab, *_ = predict_strategy(cal, lab, tst)
    sub = tab[["batch_id"]].assign(project="B", label=tab["label"].values)
    print(f"[Process B] Test predictions: {sub.label.value_counts().to_dict()}")
    print(f"[Process B] Rejected: {', '.join(sub.loc[sub.label == 'rejected', 'batch_id'])}")
    if out_path:
        sub.to_csv(out_path, index=False)
        print(f"[Process B] written -> {out_path}")
    return sub


# --------------------------------------------------------- robustness experiments
def _f1(y, p):
    return f1_score(y, p, zero_division=0)


def learning_curve(df_calib, df_labels, sizes=((5, 1), (5, 2), (10, 2), (15, 3), (20, 3), (28, 4)),
                   n_rep=100, seed=0):
    """Random small calibration sets -> held-out F1 of classifier and of MSPC (MSPC uses only the
    accepted part of the subset)."""
    ids, A = unfold(df_calib, [c for c in df_calib.columns if c != "batch_id"])
    lab = df_labels.set_index("batch_id").loc[ids, "label"]
    y = (lab == "rejected").astype(int).values
    X6, _, _, _ = prepare_data(df_calib, df_labels, TOP3_SENSORS)  # same sorted order
    rng = np.random.default_rng(seed)
    acc, rej = np.where(y == 0)[0], np.where(y == 1)[0]
    rows = []
    mon = BatchMSPC()
    Xf = mon._features(A)
    level_scale = mon._floor(A)
    for na, nr in sizes:
        f_clf, f_mspc, fa_mspc = [], [], []
        for _ in range(n_rep):
            tr = np.r_[rng.choice(acc, na, False), rng.choice(rej, nr, False)]
            te = np.setdiff1d(np.arange(len(y)), tr)
            pc = make_pipeline(make_lr()).fit(X6[tr], y[tr]).predict(X6[te])
            f_clf.append(_f1(y[te], pc))
            # MSPC on accepted subset only, LOO limits
            tra = tr[y[tr] == 0]
            core = mon._fit_core(Xf[tra], mon.var, level_scale)
            loo = [mon._score_core(mon._fit_core(np.delete(Xf[tra], i, 0), mon.var, level_scale),
                                   Xf[tra][[i]])[1][0] for i in range(len(tra))]
            m, v = np.mean(loo), np.var(loo) + 1e-12
            lim = (v / (2 * m)) * sps.chi2.ppf(0.99, 2 * m ** 2 / v)
            pm = (mon._score_core(core, Xf[te])[1] > lim).astype(int)
            f_mspc.append(_f1(y[te], pm))
            fa_mspc.append(pm[y[te] == 0].mean())
        rows.append({"n_accepted": na, "n_rejected": nr,
                     "clf_F1_mean": np.mean(f_clf), "clf_F1_p10": np.percentile(f_clf, 10),
                     "mspc_F1_mean": np.mean(f_mspc), "mspc_F1_p10": np.percentile(f_mspc, 10),
                     "mspc_false_alarm_rate": np.mean(fa_mspc)})
    return pd.DataFrame(rows)


def leave_one_rejected_out(df_calib, df_labels):
    """Train the classifier without one rejected batch, score that batch (unseen-fault proxy)."""
    rej = df_labels.loc[df_labels.label == "rejected", "batch_id"].tolist()
    acc_ids = df_labels.loc[df_labels.label == "accepted", "batch_id"]
    mspc = BatchMSPC().fit(df_calib[df_calib.batch_id.isin(acc_ids)])
    rows = []
    for b in rej:
        keep = df_labels.batch_id != b
        pipe, cols = fit_classifier(df_calib[df_calib.batch_id != b], df_labels[keep])
        held = df_calib[df_calib.batch_id == b]
        s = mspc.score(held).iloc[0]
        rows.append({"held_out": b, "clf_p_rejected": classifier_proba(pipe, cols, held).iloc[0],
                     "mspc_SPE_ratio": s.SPE_ratio, "mspc_alarm": bool(s.alarm)})
    return pd.DataFrame(rows)


def inject(df_batch: pd.DataFrame, sensor: str, kind: str, size: float, window=(0.10, 0.25)):
    """Synthetic disturbance on one batch.
    kind: 'none' (baseline false-alarm check) | 'step' (+size × level in window) | 'level' (whole-batch offset, benign set-point change)
          | 'noise' (gaussian noise, sd = size × level)."""
    d = df_batch.copy()
    n = len(d)
    j = d.columns.get_loc(sensor)
    lvl = float(np.median(np.abs(d[sensor]))) or 1.0
    if kind == "none":
        pass
    elif kind == "step":
        a, b = int(window[0] * n), int(window[1] * n)
        d.iloc[a:b, j] = d.iloc[a:b, j] + size * lvl
    elif kind == "level":
        d.iloc[:, j] = d.iloc[:, j] + size * lvl
    elif kind == "noise":
        d.iloc[:, j] = d.iloc[:, j] + np.random.default_rng(0).normal(0, size * lvl, n)
    return d


def disturbance_test(df_calib, df_labels, scenarios, max_batches=None):
    """For each scenario (sensor, kind, size) inject into every accepted batch (held out of BOTH
    models) and report the fraction flagged by classifier and by MSPC."""
    acc = df_labels.loc[df_labels.label == "accepted", "batch_id"].tolist()[:max_batches]
    flags = {sc: [0, 0] for sc in scenarios}
    for b in acc:
        keep = df_labels.batch_id != b
        cal_b = df_calib[df_calib.batch_id != b]
        pipe, cols = fit_classifier(cal_b, df_labels[keep])
        ref = cal_b[cal_b.batch_id.isin([a for a in acc if a != b])]
        mspc = BatchMSPC().fit(ref)
        held = df_calib[df_calib.batch_id == b]
        for sc in scenarios:
            d = inject(held, *sc)
            flags[sc][0] += classifier_proba(pipe, cols, d).iloc[0] > 0.5
            flags[sc][1] += bool(mspc.score(d).alarm.iloc[0])
    return pd.DataFrame([{"sensor": s, "kind": k, "size": z,
                          "classifier_flagged": f"{v[0]}/{len(acc)}", "mspc_flagged": f"{v[1]}/{len(acc)}"}
                         for (s, k, z), v in flags.items()])


# ------------------------------------------------------------ self-training update
def pseudo_label(pipe, cols, mspc, df_new, p_low=0.05, p_high=0.5, accept_margin=5.0):
    """Confident pseudo-labels for unlabelled new batches.
    accepted : classifier p < p_low AND MSPC statistics < accept_margin x control limit
               (a mild excursion on a model built from few batches is usually un-modelled normal
               variability; real faults sit hundreds of times above the limit)
    rejected : classifier p > p_high AND MSPC alarm (both layers agree)
    everything else -> left unlabelled, routed to human review."""
    tab = mspc.score(df_new)
    tab["p"] = classifier_proba(pipe, cols, df_new).loc[tab.batch_id].values
    worst = tab[["T2_ratio", "SPE_ratio"]].max(1)
    acc = tab.loc[(tab.p < p_low) & (worst < accept_margin), "batch_id"].tolist()
    rej = tab.loc[(tab.p > p_high) & tab.alarm, "batch_id"].tolist()
    return acc, rej


def self_training_update(df_calib, df_labels, df_new, rounds=1, accept_margin=5.0, verbose=True):
    """Fold confidently pseudo-labelled new batches into the reference set and refit both layers."""
    cal, lab = df_calib.copy(), df_labels.copy()
    history = []
    for r in range(rounds + 1):
        pipe, cols = fit_classifier(cal, lab)
        mspc = BatchMSPC().fit(cal[cal.batch_id.isin(lab.loc[lab.label == "accepted", "batch_id"])])
        history.append({"round": r, "n_accepted_ref": int((lab.label == "accepted").sum()),
                        "n_rejected_ref": int((lab.label == "rejected").sum()),
                        "n_PCs": mspc.n_pc, "T2_lim": mspc.t2_lim, "SPE_lim": mspc.spe_lim})
        if r == rounds:
            break
        pa, pr = pseudo_label(pipe, cols, mspc, df_new, accept_margin=accept_margin)
        new = [b for b in pa + pr if b not in set(lab.batch_id)]
        if not new:
            break
        add = pd.DataFrame({"batch_id": new, "label": ["accepted" if b in pa else "rejected" for b in new]})
        lab = pd.concat([lab, add], ignore_index=True)
        cal = pd.concat([cal, df_new[df_new.batch_id.isin(new)]], ignore_index=True)
    hist = pd.DataFrame(history)
    if verbose:
        print(hist.round(2).to_string(index=False))
    return cal, lab, hist


def self_training_benefit(df_calib, df_labels, df_pool, n_acc=10, n_rej=2, n_rep=30, rounds=2,
                          accept_margin=5.0, seed=1):
    """Start from a SMALL labelled set, self-train on the unlabelled pool (the test batches), and
    evaluate on the held-out labelled calibration batches: before vs after the update."""
    rng = np.random.default_rng(seed)
    acc = df_labels.loc[df_labels.label == "accepted", "batch_id"].values
    rej = df_labels.loc[df_labels.label == "rejected", "batch_id"].values
    rows = []
    for _ in range(n_rep):
        tr = list(rng.choice(acc, n_acc, False)) + list(rng.choice(rej, n_rej, False))
        lab_s, cal_s = df_labels[df_labels.batch_id.isin(tr)], df_calib[df_calib.batch_id.isin(tr)]
        held = df_labels[~df_labels.batch_id.isin(tr)]
        df_held = df_calib[df_calib.batch_id.isin(held.batch_id)]
        y = (held.set_index("batch_id").label == "rejected").astype(int)

        def evaluate(cal_x, lab_x):
            pipe, cols = fit_classifier(cal_x, lab_x)
            mspc = BatchMSPC().fit(cal_x[cal_x.batch_id.isin(lab_x.loc[lab_x.label == "accepted", "batch_id"])])
            p = classifier_proba(pipe, cols, df_held)
            al = mspc.score(df_held).set_index("batch_id").alarm.loc[p.index]
            yy = y.loc[p.index]
            return {"clf_F1": _f1(yy, (p > 0.5).astype(int)),
                    "mspc_recall": al[yy == 1].mean(), "mspc_false_alarm": al[yy == 0].mean(),
                    "combined_F1": _f1(yy, ((p > 0.5) | al).astype(int)), "SPE_lim": mspc.spe_lim}

        b = evaluate(cal_s, lab_s)
        cal_u, lab_u, _ = self_training_update(cal_s, lab_s, df_pool, rounds=rounds,
                                               accept_margin=accept_margin, verbose=False)
        a = evaluate(cal_u, lab_u)
        rows.append({**{f"{k}_before": v for k, v in b.items()}, **{f"{k}_after": v for k, v in a.items()},
                     "n_pseudo_labelled": len(lab_u) - len(lab_s),
                     "pseudo_rejected": int((lab_u.label == "rejected").sum() - (lab_s.label == "rejected").sum())})
    return pd.DataFrame(rows)
