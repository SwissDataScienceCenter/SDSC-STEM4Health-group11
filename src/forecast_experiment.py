"""Compare accepted-only one-step forecasting features for Process A."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, f1_score, roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

if __package__:
    from . import brpmm
else:
    import brpmm


def lagged(batch, history=8):
    """Predict the next increment using the increments between the preceding readings."""
    changes = np.diff(batch, axis=0)
    if len(changes) < history:
        return np.empty((batch.shape[1], 0, history)), np.empty((batch.shape[1], 0))
    windows = np.lib.stride_tricks.sliding_window_view(changes, history, axis=0)
    design = np.concatenate((np.ones((*windows.shape[:2], 1)), windows[..., :-1]), axis=2)
    return design.transpose(1, 0, 2).copy(), windows[..., -1].T.copy()


def moments(design, target):
    transposed = design.transpose(0, 2, 1)
    count = max(1, target.shape[1])
    return transposed @ design / count, (transposed @ target[..., None]) / count


def summarize(errors):
    absolute = np.abs(errors)
    return np.column_stack((errors.mean(axis=1), absolute.mean(axis=1),
                            np.quantile(absolute, .95, axis=1))).ravel()


class ForecastFeatures:
    def __init__(self, batches, labels, history=8):
        self.data = [lagged(batch, history) for batch in batches]
        self.labels = labels
        self.valid = np.array([target.shape[1] > 0 for _, target in self.data])
        self.gram, self.cross = map(np.stack, zip(*(moments(*item) for item in self.data)))
        self.persistence = {i: summarize(target) for i, (_, target) in enumerate(self.data)
                            if self.valid[i]}

    def fit(self, train):
        accepted = train[(self.labels[train] == 0) & self.valid[train]]
        if not len(accepted):
            raise ValueError("Forecasting requires an accepted training batch with enough readings for the forecast history")
        gram = self.gram[accepted].mean(axis=0)
        # Scale the ridge penalty to the typical squared increment of each sensor.
        scale = np.maximum(np.diagonal(gram, axis1=1, axis2=2)[:, 1:].mean(axis=1), 1e-12)
        penalty = np.diag([0., *([.1] * (gram.shape[-1] - 1))])
        coef = np.linalg.solve(gram + scale[:, None, None] * penalty, self.cross[accepted].mean(axis=0))
        return coef, accepted

    def residual(self, index, coef):
        design, target = self.data[index]
        predicted = (design @ coef).squeeze(-1)
        return summarize(target - predicted)

    def transform(self, indices, fitted):
        coef, accepted = fitted
        fallback = persistence_fallback = None
        if not self.valid[indices].all():
            fallback = np.median([self.residual(i, coef) for i in accepted], axis=0)
            persistence_fallback = np.median([self.persistence[i] for i in accepted], axis=0)
        learned = [self.residual(i, coef) if self.valid[i] else fallback for i in indices]
        persistence = [self.persistence.get(i, persistence_fallback) for i in indices]
        return np.stack(persistence), np.stack(learned)

    def crossfit(self, train):
        result = np.empty((2, len(train), self.gram.shape[1] * 3))
        cv = StratifiedKFold(3, shuffle=True, random_state=0)
        for fit, hold in cv.split(train, self.labels[train]):
            result[:, hold] = self.transform(train[hold], self.fit(train[fit]))
        return result


def scores(summary, sequence, labels, forecasters, train, hold, residual_only=False):
    model = brpmm.ProcessAModel(brpmm.CFG["A"])
    model.classifier.set_params(n_jobs=1)
    model.fit(summary[train], sequence[train], labels[train])
    baseline = model.score(summary[hold], sequence[hold])
    x = model._classifier_inputs(summary[train], sequence[train])
    xt = model._classifier_inputs(summary[hold], sequence[hold])
    flag = (model._envelope_shifts(summary[hold]) > model.envelope_limits).any(axis=1)
    train_errors = forecasters.crossfit(train)
    test_errors = forecasters.transform(hold, forecasters.fit(train))
    result = [baseline]
    for a, c in zip(train_errors, test_errors):
        if residual_only:
            model.classifier.fit(a, labels[train])
            result.append(model.classifier.predict_proba(c)[:, 1])
        else:
            model.classifier.fit(np.column_stack((x, a)), labels[train])
            result.append(model.classifier.predict_proba(np.column_stack((xt, c)))[:, 1] + flag)
    return np.stack(result)


def evaluate_fold(summary, sequence, labels, forecasters, train, hold, residual_only=False):
    inner = np.zeros((3, len(train)))
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0)
    for fit, val in cv.split(train, labels[train]):
        inner[:, val] += scores(summary, sequence, labels, forecasters, train[fit], train[val], residual_only) / 3
    held = scores(summary, sequence, labels, forecasters, train, hold, residual_only)
    cuts = np.array([brpmm.process_a_threshold(labels[train], row) for row in inner])
    return labels[hold], held, held >= cuts[:, None]


def report(name, results, budget):
    y, scores_, predictions = [np.concatenate(parts, axis=-1) for parts in zip(*results)]
    for variant, s, p in zip(("baseline", "persistence", "learned"), scores_, predictions):
        fpr, tpr, _ = roc_curve(y, s, drop_intermediate=False)
        caught = round(tpr[np.rint(fpr * (y == 0).sum()) <= budget].max() * y.sum())
        print(name, variant, f"accuracy={accuracy_score(y, p):.4f}",
              f"accepted_F1={f1_score(y, p, pos_label=0):.4f}",
              f"rejected_F1={f1_score(y, p):.4f}",
              f"rejects={sum(p & (y == 1))}/{sum(y)}",
              f"false_alarms={sum(p & (y == 0))}", f"rejects_at_{budget}_FP={caught}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("../input-data/Project 2 - BRPMM"))
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--history", type=int, choices=(4, 8, 16), nargs="+", default=[8])
    parser.add_argument("--residual-only", action="store_true",
                        help="Use residual summaries alone for the two forecast variants")
    args = parser.parse_args()
    root = args.data / "process A" / "calibration data"
    frame = pd.read_csv(root / "calibration_data.csv")
    ids, summary, sequence = brpmm.featurise(frame, brpmm.CFG["A"])
    labels = brpmm.labels_to_y(pd.read_csv(root / "calibration_labels.csv"), ids)
    columns = frame.columns.drop("batch_id")
    batches = [group[columns].to_numpy(float) for _, group in frame.groupby("batch_id", sort=False)]
    if len(columns) != 11:
        raise ValueError("Expected 11 Process A sensors")
    print(f"Prepared {len(ids)} calibration batches", flush=True)
    # Build the lag representation once per repeat worker.
    def repeat(seed, history):
        forecasters = ForecastFeatures(batches, labels, history)
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        results = [evaluate_fold(summary, sequence, labels, forecasters, tr, te, args.residual_only)
                   for tr, te in cv.split(summary, labels)]
        report(f"history={history} seed={seed}", results, 12)
        return results

    with Parallel(n_jobs=args.jobs) as parallel:
        for history in args.history:
            results = parallel(delayed(repeat)(seed, history) for seed in (307, 359, 401, 463))
            report(f"history={history} pooled", [fold for run in results for fold in run], 50)


if __name__ == "__main__":
    main()
