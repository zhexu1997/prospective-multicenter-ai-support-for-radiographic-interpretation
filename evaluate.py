#!/usr/bin/env python3
"""Score locked AI suggestions against the 60-case reference diagnoses."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from matching import is_correct

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=DATA / "ai_suggestions.csv",
        help="CSV with case_id, model, primary_diagnosis_zh (locked study outputs or run_api.py results)",
    )
    args = parser.parse_args()

    if "primary_diagnosis_zh" not in ai.columns and "primary_diagnosis" in ai.columns:
        ai["primary_diagnosis_zh"] = ai["primary_diagnosis"]
    df = ai.merge(
        cases[["case_id", "body_region", "reference_diagnosis_zh", "reference_diagnosis_en"]],
        on="case_id",
        how="left",
    )
    df["correct"] = [
        is_correct(pred, gold)
        for pred, gold in zip(df["primary_diagnosis_zh"], df["reference_diagnosis_zh"])
    ]

    by_model = (
        df.groupby("model", sort=True)
        .agg(n_cases=("case_id", "nunique"), n_correct=("correct", "sum"), accuracy=("correct", "mean"))
        .reset_index()
    )
    by_model["accuracy"] = (by_model["accuracy"] * 100).round(1)
    by_model["n_correct"] = by_model["n_correct"].astype(int)

    by_region = (
        df.groupby(["model", "body_region"], sort=True)
        .agg(n_cases=("case_id", "nunique"), n_correct=("correct", "sum"), accuracy=("correct", "mean"))
        .reset_index()
    )
    by_region["accuracy"] = (by_region["accuracy"] * 100).round(1)
    by_region["n_correct"] = by_region["n_correct"].astype(int)

    print(f"Cases: {cases['case_id'].nunique()}")
    print(f"Predictions: {args.predictions}")
    print("\nTop-1 accuracy")
    print(by_model.to_string(index=False))
    print("\nBy body region")
    print(by_region.to_string(index=False))

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    case_scores = df[
        [
            "case_id",
            "model",
            "reference_diagnosis_en",
            "primary_diagnosis_zh",
            "correct",
        ]
    ].sort_values(["case_id", "model"])
    case_scores.to_csv(out / "ai_case_scores.csv", index=False, encoding="utf-8-sig")
    by_model.to_csv(out / "ai_accuracy.csv", index=False, encoding="utf-8-sig")
    print(f"\nWrote {out / 'ai_accuracy.csv'} and {out / 'ai_case_scores.csv'}")


if __name__ == "__main__":
    main()
