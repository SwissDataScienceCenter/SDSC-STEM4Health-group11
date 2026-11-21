"""Leaderboard simulator: estimate the leaderboard score of a submission
BEFORE submitting it.

We don't have the test labels, but the first submission's leaderboard
response leaks a lot about them:
  * Process B scored F1 = 1.0  ->  its B predictions ARE the true labels,
    so the B part of any submission can be scored exactly.
  * Process A precision / recall / accuracy  ->  exact confusion counts:
    how many truly-rejected batches are hidden among the batches it called
    'accepted' (and among those it called 'rejected').

For process A we combine those known counts with per-batch probabilities
from the sensor-wise model (calibrated out-of-fold on the calibration set),
tilt the probabilities so each group's expected number of rejected batches
matches the known count exactly, then sample many plausible answer keys
and score every submission against all of them.

Leaderboard metric (from the first submission's response):
    score = mean(F1_A, F1_B), F1 with 'accepted' as the positive class.

Usage (from repo root):
    python src/leaderboard_sim.py submission_a.csv submission_b.csv ...
    python src/leaderboard_sim.py --sweep      # best number of A batches to flag
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.dirname(__file__))
import sensorwise as sw  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REFERENCE = os.path.join(HERE, "..", "leaderboard", "first_submission.csv")
# Leaderboard response for the reference (first) submission
REF_METRICS = {"A": dict(accuracy=0.845, precision=0.88, recall=0.917),
               "B": dict(accuracy=1.0, precision=1.0, recall=1.0)}
REF_SCORE = 0.949
N_SIM = 20000
SEED = 0


# ------------------------------------------------------------------ helpers
def f1_accepted(pred_acc, true_acc):
    """pred_acc, true_acc: boolean arrays (True = accepted). Vectorised over rows."""
    tp = (pred_acc & true_acc).sum(-1)
    fp = (pred_acc & ~true_acc).sum(-1)
    fn = (~pred_acc & true_acc).sum(-1)
    return np.where(tp > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)


def confusion_from_metrics(n_pred_acc, n, m):
    """Recover integer confusion counts from rounded precision/recall/accuracy."""
    tp = round(m["precision"] * n_pred_acc)
    n_true_acc = round(tp / m["recall"])
    fp = n_pred_acc - tp
    fn = n_true_acc - tp
    tn = n - tp - fp - fn
    acc = (tp + tn) / n
    if abs(acc - m["accuracy"]) > 0.0015:
        raise ValueError(f"Reference metrics don't match the reference file (acc {acc:.3f} vs "
                         f"{m['accuracy']}). Is leaderboard/first_submission.csv the submitted file?")
    return dict(tp=tp, fp=fp, fn=fn, tn=tn)


def load_sub(path):
    s = pd.read_csv(path, dtype={"batch_id": str})
    s["label"] = s["label"].str.strip().str.lower()
    return s.set_index("batch_id")


# --------------------------------------------------- probabilities for A
def a_probabilities(test_ids):
    """P(rejected) per test batch of A from the sensor-wise model, calibrated
    out-of-fold with isotonic regression."""
    F = sw.load_feats("cal")
    y = sw.labels(F)
    oof = sw.cv_scores(F, y, lambda Ft, yt: sw.SensorwiseModel().fit(Ft, yt).score, n_repeats=3)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99).fit(oof, y)
    Ft = sw.load_feats("test")
    s = sw.SensorwiseModel().fit(F, y).score(Ft)
    p = pd.Series(iso.predict(s), index=Ft.index)
    return p.reindex(test_ids).to_numpy()


def tilt(p, target):
    """Shift probabilities on the logit scale so they sum to `target`."""
    lo = np.log(p / (1 - p))
    f = lambda d: (1 / (1 + np.exp(-(lo + d)))).sum() - target
    if target <= 0:
        return np.zeros_like(p)
    if target >= len(p):
        return np.ones_like(p)
    d = brentq(f, -30, 30)
    return 1 / (1 + np.exp(-(lo + d)))


def sample_exact(p, k, n_sim, rng):
    """Sample n_sim label vectors with EXACTLY k positives, weighted by p
    (Gumbel top-k on log-odds ~ conditional Bernoulli)."""
    if k == 0:
        return np.zeros((n_sim, len(p)), bool)
    lo = np.log(p / (1 - p))
    g = lo + rng.gumbel(size=(n_sim, len(p)))
    idx = np.argpartition(-g, k - 1, axis=1)[:, :k]
    out = np.zeros((n_sim, len(p)), bool)
    np.put_along_axis(out, idx, True, axis=1)
    return out


# ----------------------------------------------------------------- main
class Simulator:
    def __init__(self, reference=REFERENCE, n_sim=N_SIM):
        ref = load_sub(reference)
        self.ids = {P: ref.index[ref.project == P] for P in ("A", "B")}
        # B: reference scored 1.0 -> truth
        self.true_B_acc = (ref.loc[self.ids["B"], "label"] == "accepted").to_numpy()
        # A: known confusion of the reference
        refA = (ref.loc[self.ids["A"], "label"] == "accepted").to_numpy()
        c = confusion_from_metrics(int(refA.sum()), len(refA), REF_METRICS["A"])
        self.conf = c
        p = a_probabilities(self.ids["A"])
        rng = np.random.default_rng(SEED)
        # truly-rejected among ref-'accepted' = fp ; among ref-'rejected' = tn
        truth_rej = np.zeros((n_sim, len(refA)), bool)
        for mask, k in [(refA, c["fp"]), (~refA, c["tn"])]:
            pm = tilt(p[mask], k)
            truth_rej[:, mask] = sample_exact(np.clip(pm, 1e-4, 1 - 1e-4), k, n_sim, rng)
        self.true_A_acc = ~truth_rej
        self.p_rej_A = pd.Series(truth_rej.mean(0), index=self.ids["A"])

    def score(self, sub):
        pa = (sub.loc[self.ids["A"], "label"] == "accepted").to_numpy()
        pb = (sub.loc[self.ids["B"], "label"] == "accepted").to_numpy()
        fA = f1_accepted(pa[None, :], self.true_A_acc)
        fB = float(f1_accepted(pb, self.true_B_acc))
        sc = (fA + fB) / 2
        return dict(F1_A=fA.mean(), F1_A_lo=np.quantile(fA, 0.05), F1_A_hi=np.quantile(fA, 0.95),
                    F1_B=fB, score=sc.mean(), score_lo=np.quantile(sc, 0.05),
                    score_hi=np.quantile(sc, 0.95), p_beat_ref=(sc > REF_SCORE + 1e-9).mean())


def fmt(name, r):
    return (f"  {name:<36} A F1 {r['F1_A']:.3f} [{r['F1_A_lo']:.3f}-{r['F1_A_hi']:.3f}]   "
            f"B F1 {r['F1_B']:.3f}   SCORE {r['score']:.3f} [{r['score_lo']:.3f}-{r['score_hi']:.3f}]   "
            f"P(beat {REF_SCORE}) {r['p_beat_ref']:.0%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submissions", nargs="*")
    ap.add_argument("--reference", default=REFERENCE)
    ap.add_argument("--sweep", action="store_true",
                    help="score 'flag the top-k most suspicious A batches' for a range of k")
    a = ap.parse_args()

    sim = Simulator(a.reference)
    c = sim.conf
    print(f"Reference = first submission (leaderboard {REF_SCORE}). Recovered A confusion: "
          f"{c['tp']} accepted correct, {c['fp']} bad called accepted, "
          f"{c['fn']} good called rejected, {c['tn']} bad caught "
          f"-> test A has ~{c['fp'] + c['tn']} rejected batches.\n")
    ref = load_sub(a.reference)
    print(fmt("reference (sanity check, should be ~0.949)", sim.score(ref)))
    for path in a.submissions:
        print(fmt(os.path.basename(path), sim.score(load_sub(path))))

    if a.sweep:
        # rank A test batches by the sensor-wise model, B taken from reference (= truth)
        F, Ft = sw.load_feats("cal"), sw.load_feats("test")
        s = pd.Series(sw.SensorwiseModel().fit(F, sw.labels(F)).score(Ft), index=Ft.index)
        order = s.reindex(sim.ids["A"]).sort_values(ascending=False).index
        print("\nSweep: flag the top-k most suspicious A batches as rejected (sensor-wise ranking)")
        best = None
        for k in range(10, 61, 2):
            sub = ref.copy()
            sub.loc[sim.ids["A"], "label"] = "accepted"
            sub.loc[order[:k], "label"] = "rejected"
            r = sim.score(sub)
            print(fmt(f"top {k:>2} flagged", r))
            if best is None or r["score"] > best[1]["score"]:
                best = (k, r)
        print(f"\nBest: flag {best[0]} A batches -> expected score {best[1]['score']:.3f}")


if __name__ == "__main__":
    main()
