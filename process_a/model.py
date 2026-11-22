"""Train or apply the Process A temperature-forecast classifier."""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from torch import nn

SEED = 0
CONFIG = {"n_bins": 30, "use_length": True}


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
    targeted_sensors,
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
    traj.append(_rolling_mad(X).max(axis=0))
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
    return np.array(summ, float), np.concatenate(traj)


def featurise(df: pd.DataFrame, cfg: dict):
    cols = df.columns.drop("batch_id").tolist()
    targeted_sensors = tuple(cols.index(name) for name in ("temperature_1", "pressure_4", "density"))
    ids, S, Tr = [], [], []
    for bid, g in df.groupby("batch_id", sort=False):
        s, tr = batch_features(
            g[cols].to_numpy(float), cfg["n_bins"], cfg["use_length"],
            targeted_sensors=targeted_sensors,
        )
        ids.append(bid), S.append(s), Tr.append(tr)
    return ids, np.vstack(S), np.vstack(Tr)


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
        targeted = trajectories[:, end_maxima:end_maxima + 3].copy()
        targeted[:, 2] -= self.density_centers[mode]
        inputs.append(targeted)
        return np.column_stack(inputs)

    def _envelope_long_mode(self, summaries):
        return summaries[:, -1] >= np.log(self.ENVELOPE_MODE_LENGTH)

    def _envelope_shifts(self, summaries):
        long_mode = self._envelope_long_mode(summaries)
        centers = self.envelope_centers[long_mode.astype(int)]
        return np.abs(summaries[:, self.SENSOR_MEDIANS] - centers)

    def __init__(self, cfg):
        self.n_bins = cfg["n_bins"]
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
        density = Tr[y == 0, maxima_start + n_sensors + 2]
        self.density_centers = np.array([
            np.median(density[envelope_long == mode]) if (envelope_long == mode).any() else np.median(density)
            for mode in (False, True)
        ])
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


TEMPERATURES = tuple(f"temperature_{i}" for i in range(1, 5))
HISTORY, HORIZON = 16, 8


def windows(batch, count):
    last = len(batch) - HISTORY - HORIZON
    if last < 0:
        raise ValueError("The temperature GRU needs at least 24 readings per batch")
    starts = np.linspace(0, last, min(count, last + 1), dtype=int)
    past = np.lib.stride_tricks.sliding_window_view(batch, HISTORY, axis=0)[starts]
    future = batch[starts[:, None] + np.arange(HISTORY, HISTORY + HORIZON)].transpose(0, 2, 1)
    return past, future - past[:, :, -1, None]


def batch_windows(batches, count):
    parts = [windows(batch, count) for batch in batches]
    past, future = map(np.concatenate, zip(*parts))
    lengths = [len(x) for x, _ in parts]
    modes = np.repeat([len(batch) >= ProcessAModel.ENVELOPE_MODE_LENGTH
                       for batch in batches], lengths).astype(int)
    return past, future, modes, np.cumsum(lengths)[:-1]


def gru_inputs(past, modes, centers, scale):
    normalized = (past - centers[modes, :, None]) / scale[None, :, None]
    return torch.tensor(normalized.transpose(0, 2, 1), dtype=torch.float32)


def fit_forecaster(accepted):
    past, future, modes, _ = batch_windows(accepted, 24)
    centers = np.stack([np.median(past[modes == mode] if (modes == mode).any() else past,
                                  axis=(0, 2)) for mode in (0, 1)])
    scale = np.maximum(np.std(past - centers[modes, :, None], axis=(0, 2)), 1e-8)
    inputs = gru_inputs(past, modes, centers, scale)
    targets = torch.tensor((future / scale[None, :, None]).reshape(len(past), -1),
                           dtype=torch.float32)
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        gru = nn.GRU(4, 8, batch_first=True)
        head = nn.Linear(8, 4 * HORIZON)
        optimizer = torch.optim.AdamW([*gru.parameters(), *head.parameters()],
                                      lr=.01, weight_decay=.03)
        for _ in range(16):
            optimizer.zero_grad()
            nn.functional.smooth_l1_loss(head(gru(inputs)[1][0]), targets).backward()
            optimizer.step()
    return gru.eval(), head.eval(), centers, scale


def forecast_features(forecaster, batches):
    gru, head, centers, scale = forecaster
    past, actual, modes, boundaries = batch_windows(batches, 96)
    inputs = gru_inputs(past, modes, centers, scale)
    with torch.no_grad():
        predicted = head(gru(inputs)[1][0]).numpy().reshape(actual.shape)
    errors = np.abs(actual - predicted * scale[None, :, None])
    return np.stack([np.quantile(part[:, :2, :], .95, axis=(0, 2))
                     for part in np.split(errors, boundaries)])


def fit_classifier(summaries, trajectories, batches, y, config):
    extras = np.empty((len(y), 2))
    cv = StratifiedKFold(2, shuffle=True, random_state=0)
    for train, hold in cv.split(summaries, y):
        reference = fit_forecaster([batches[i] for i in train if y[i] == 0])
        extras[hold] = forecast_features(reference, [batches[i] for i in hold])
    model = ProcessAModel(config).fit(summaries, trajectories, y)
    model.classifier.fit(np.column_stack((model._classifier_inputs(summaries, trajectories), extras)), y)
    return model, fit_forecaster([batch for batch, label in zip(batches, y) if label == 0])


def score(model, forecaster, summaries, trajectories, batches):
    inputs = np.column_stack((model._classifier_inputs(summaries, trajectories),
                              forecast_features(forecaster, batches)))
    probability = model.classifier.predict_proba(inputs)[:, 1]
    outside = (model._envelope_shifts(summaries) > model.envelope_limits).any(axis=1)
    return probability + outside


def extract(frame, config, sensors):
    ordered = frame[["batch_id", *sensors]]
    ids, summaries, trajectories = featurise(ordered, config)
    batches = [group[list(TEMPERATURES)].to_numpy(float)
               for _, group in ordered.groupby("batch_id", sort=False)]
    return ids, summaries, trajectories, batches


def rejected_threshold(y, scores):
    p, r, cuts = precision_recall_curve(y, scores)
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=p + r > 0)[:-1]
    best = np.flatnonzero(f1 == f1.max())
    return float(cuts[best[len(best) // 2]])


def fit_model(calibration, labels):
    config = CONFIG.copy()
    sensors = calibration.columns.drop("batch_id").tolist()
    ids, summaries, trajectories, batches = extract(calibration, config, sensors)
    y = labels_to_y(labels, ids)
    oof = np.empty(len(y))
    for train, hold in StratifiedKFold(3, shuffle=True, random_state=0).split(summaries, y):
        model, forecaster = fit_classifier(summaries[train], trajectories[train],
                                           [batches[i] for i in train], y[train], config)
        oof[hold] = score(model, forecaster, summaries[hold], trajectories[hold],
                          [batches[i] for i in hold])
    threshold = rejected_threshold(y, oof)
    model, forecaster = fit_classifier(summaries, trajectories, batches, y, config)
    return dict(model=model, forecaster=forecaster, threshold=threshold,
                config=config, sensor_columns=sensors, n_features=95)


def predict(artifact, frame):
    ids, summaries, trajectories, batches = extract(frame, artifact["config"], artifact["sensor_columns"])
    scores = score(artifact["model"], artifact["forecaster"], summaries, trajectories, batches)
    return pd.DataFrame(dict(batch_id=ids, score=scores,
                             label=np.where(scores >= artifact["threshold"], "rejected", "accepted")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("process_a_95.joblib"))
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--data", type=Path, default=Path("../input-data/Project 2 - BRPMM"))
    inference = commands.add_parser("predict")
    inference.add_argument("--input", type=Path, required=True)
    inference.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "train":
        root = args.data / "process A" / "calibration data"
        artifact = fit_model(pd.read_csv(root / "calibration_data.csv"),
                             pd.read_csv(root / "calibration_labels.csv"))
        args.model.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, args.model, compress=3)
        print(f"Saved {args.model} with rejection cutoff {artifact['threshold']:.6f}")
    else:
        predictions = predict(joblib.load(args.model), pd.read_csv(args.input))
        predictions.to_csv(args.out, index=False)


if __name__ == "__main__":
    # Keep the model class importable by its package name when reloading joblib.
    from process_a.model import main as package_main
    package_main()
