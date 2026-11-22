"""Train or apply the Process A temperature-forecast classifier."""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from torch import nn

if __package__:
    from . import brpmm
else:
    import brpmm

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
    modes = np.repeat([len(batch) >= brpmm.ProcessAModel.ENVELOPE_MODE_LENGTH
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
    model = brpmm.ProcessAModel(config).fit(summaries, trajectories, y)
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
    ids, summaries, trajectories = brpmm.featurise(ordered, config)
    batches = [group[list(TEMPERATURES)].to_numpy(float)
               for _, group in ordered.groupby("batch_id", sort=False)]
    return ids, summaries, trajectories, batches


def rejected_threshold(y, scores):
    p, r, cuts = precision_recall_curve(y, scores)
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=p + r > 0)[:-1]
    best = np.flatnonzero(f1 == f1.max())
    return float(cuts[best[len(best) // 2]])


def fit_model(calibration, labels):
    config = brpmm.CFG["A"].copy()
    sensors = calibration.columns.drop("batch_id").tolist()
    ids, summaries, trajectories, batches = extract(calibration, config, sensors)
    y = brpmm.labels_to_y(labels, ids)
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


def predict(artifact, frame, rate=None):
    ids, summaries, trajectories, batches = extract(frame, artifact["config"], artifact["sensor_columns"])
    scores = score(artifact["model"], artifact["forecaster"], summaries, trajectories, batches)
    if rate is None:
        cut = artifact["threshold"]
    else:  # flag the k highest-scoring batches, k = round(rate * n)
        k = max(1, int(round(rate * len(scores))))
        cut = np.sort(scores)[::-1][k - 1]
    out = pd.DataFrame(dict(batch_id=ids, score=scores,
                            label=np.where(scores >= cut, "rejected", "accepted")))
    print(f"flagged {(out.label == 'rejected').sum()}/{len(out)} as rejected; "
          f"{int((scores >= 1).sum())} batches outside sensor limits (score>=1)")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/process_a_95.joblib"))
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--data", type=Path, default=Path("../input-data/Project 2 - BRPMM"))
    inference = commands.add_parser("predict")
    inference.add_argument("--input", type=Path, required=True)
    inference.add_argument("--out", type=Path, required=True)
    inference.add_argument("--rate", type=float, default=None,
                           help="flag this fraction of batches as rejected (e.g. 0.255); default: saved cutoff")
    args = parser.parse_args()
    if args.command == "train":
        root = args.data / "process A" / "calibration data"
        artifact = fit_model(pd.read_csv(root / "calibration_data.csv"),
                             pd.read_csv(root / "calibration_labels.csv"))
        args.model.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, args.model, compress=3)
        print(f"Saved {args.model} with rejection cutoff {artifact['threshold']:.6f}")
    else:
        predictions = predict(joblib.load(args.model), pd.read_csv(args.input), args.rate)
        predictions.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
