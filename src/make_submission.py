"""Combine Process A and B prediction CSVs in submission-template order."""
import argparse
from pathlib import Path

import pandas as pd


def make_submission(template, predictions_a, predictions_b):
    """Validate label coverage and return the original template with labels."""
    labels = []
    for process, predictions in (("A", predictions_a), ("B", predictions_b)):
        expected = template.loc[template["project"] == process, "batch_id"]
        if (not predictions["batch_id"].is_unique
                or set(predictions["batch_id"]) != set(expected)):
            raise ValueError(f"Process {process} predictions must cover each template batch once")
        if not predictions["label"].isin(["accepted", "rejected"]).all():
            raise ValueError(f"Process {process} contains an invalid label")
        labels.append(predictions.set_index("batch_id")["label"])
    return template.assign(label=template["batch_id"].map(pd.concat(labels, verify_integrity=True)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True, help="Process A prediction CSV")
    parser.add_argument("--b", type=Path, required=True, help="Process B prediction CSV")
    parser.add_argument("--data", type=Path, default=Path("../input-data/Project 2 - BRPMM"))
    parser.add_argument("--out", type=Path, default=Path("submission.csv"))
    args = parser.parse_args()
    result = make_submission(pd.read_csv(args.data / "BRPMM_template.csv"),
                             pd.read_csv(args.a), pd.read_csv(args.b))
    result.to_csv(args.out, index=False)
    print(f"Wrote {len(result)} labels to {args.out}")


if __name__ == "__main__":
    main()
