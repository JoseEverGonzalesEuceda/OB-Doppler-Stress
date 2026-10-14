from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from ob_doppler import corrected_repeated_split_test


def sign_averaged_ba(metrics: pd.DataFrame) -> pd.DataFrame:
    keys = [c for c in ["model", "strategy", "train_strategy", "test_operator", "seed", "abs_velocity_kms"] if c in metrics.columns]
    return metrics.groupby(keys, as_index=False)["balanced_accuracy"].mean()


def main_corrected_effects(metrics: pd.DataFrame, abs_velocity: float = 200.0) -> pd.DataFrame:
    q = sign_averaged_ba(metrics)
    q = q[q["abs_velocity_kms"] == abs_velocity]
    pivot = q.pivot_table(index=["model", "seed"], columns="strategy", values="balanced_accuracy").reset_index()
    rows = []
    for model, g in pivot.groupby("model"):
        if not {"raw", "raw_aug"}.issubset(g.columns):
            continue
        test = corrected_repeated_split_test((g["raw_aug"] - g["raw"]).to_numpy())
        rows.append({"model": model, "abs_velocity_kms": abs_velocity, **test})
    return pd.DataFrame(rows).sort_values("mean", ascending=False)


def control_summary(control: pd.DataFrame, abs_velocity: float = 200.0) -> pd.DataFrame:
    q = control[
        (control["test_operator"] == "linear_test")
        & (control["abs_velocity_kms"] == abs_velocity)
    ].copy()
    q = sign_averaged_ba(q)
    summary = q.groupby(["model", "train_strategy"], as_index=False)["balanced_accuracy"].agg(["mean", "std"]).reset_index()
    return summary


def xgb_cluster_bootstrap(predictions: pd.DataFrame, abs_velocity=200.0, n_boot=5000, seed=20261012):
    q = predictions[
        (predictions["model"] == "XGBoost")
        & (predictions["strategy"].isin(["raw", "raw_aug"]))
        & (predictions["abs_velocity_kms"] == abs_velocity)
    ].copy()
    keys = ["repeat", "seed", "velocity_kms", "star_name", "y_true"]
    wide = q.pivot_table(index=keys, columns="strategy", values="y_pred", aggfunc="first").dropna().reset_index()
    wide["raw_correct"] = (wide["raw"].astype(int) == wide["y_true"].astype(int)).astype(int)
    wide["aug_correct"] = (wide["raw_aug"].astype(int) == wide["y_true"].astype(int)).astype(int)
    star = wide.groupby(["star_name", "y_true"], as_index=False).agg(
        n_rows=("raw_correct", "size"),
        raw_correct=("raw_correct", "sum"),
        aug_correct=("aug_correct", "sum"),
    )

    def pooled_ba(frame, col):
        recalls = []
        for cls in (0, 1):
            g = frame[frame["y_true"] == cls]
            recalls.append(float(g[col].sum() / g["n_rows"].sum()))
        return float(np.mean(recalls))

    raw_ba = pooled_ba(star, "raw_correct")
    aug_ba = pooled_ba(star, "aug_correct")
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    by_class = {cls: star[star["y_true"] == cls].reset_index(drop=True) for cls in (0, 1)}
    for b in range(n_boot):
        recalls_raw = []
        recalls_aug = []
        for cls in (0, 1):
            g = by_class[cls]
            idx = rng.integers(0, len(g), size=len(g))
            s = g.iloc[idx]
            denom = s["n_rows"].sum()
            recalls_raw.append(float(s["raw_correct"].sum() / denom))
            recalls_aug.append(float(s["aug_correct"].sum() / denom))
        boot[b] = float(np.mean(recalls_aug) - np.mean(recalls_raw))
    return {
        "raw_ba": raw_ba,
        "raw_aug_ba": aug_ba,
        "delta_ba": aug_ba - raw_ba,
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
        "n_unique_test_stars": int(star["star_name"].nunique()),
    }


def subtype_group(value: str) -> str:
    s = str(value).strip().upper()
    match = re.match(r"^([OB])(\d+(?:\.\d+)?)", s)
    if not match:
        return ""
    cls, number = match.group(1), float(match.group(2))
    if cls == "O":
        if number < 8.5:
            return "O2-O8"
        if number < 9.5:
            return "O8.5-O9.2"
        return "O9.5-O9.7"
    if number < 1.0:
        return "B0-B0.7"
    if number <= 2.0:
        return "B1-B2"
    return "B2.5-B9.5"


def subtype_errors(predictions: pd.DataFrame, abs_velocity=200.0) -> pd.DataFrame:
    q = predictions[
        (predictions["model"] == "XGBoost")
        & (predictions["strategy"] == "raw_aug")
        & (predictions["abs_velocity_kms"].isin([0.0, abs_velocity]))
    ].copy()
    q = q[q["subtype"].astype(str).str.len() > 0]
    if q.empty:
        return pd.DataFrame()
    q["group"] = q["subtype"].map(subtype_group)
    q = q[q["group"] != ""]
    q["error"] = (q["y_true"] != q["y_pred"]).astype(float)
    per_star = q.groupby(["group", "abs_velocity_kms", "star_name"], as_index=False)["error"].mean()
    out = per_star.groupby(["group", "abs_velocity_kms"], as_index=False).agg(
        n_stars=("star_name", "nunique"),
        error_rate=("error", "mean"),
    )
    return out


def main():
    parser = argparse.ArgumentParser(description="Summarize the locally generated experiment results.")
    parser.add_argument("--main", default=Path("results/main"), type=Path)
    parser.add_argument("--control", default=Path("results/control"), type=Path)
    parser.add_argument("--bootstrap", default=5000, type=int)
    args = parser.parse_args()

    metrics = pd.read_csv(args.main / "metrics.csv")
    predictions = pd.read_csv(args.main / "predictions.csv.gz")
    control = pd.read_csv(args.control / "control_metrics.csv")

    print("\nCorrected Raw+Aug - Raw effects at |v|=200 km/s")
    print(main_corrected_effects(metrics).to_string(index=False))

    print("\nMatched-size/interpolation control at |v|=200 km/s (linear test operator)")
    print(control_summary(control).to_string(index=False))

    print("\nXGBoost star-cluster bootstrap at |v|=200 km/s")
    boot = xgb_cluster_bootstrap(predictions, n_boot=args.bootstrap)
    for key, value in boot.items():
        print(f"{key}: {value}")

    sub = subtype_errors(predictions)
    if not sub.empty:
        print("\nXGBoost Raw+Aug subtype-group errors")
        print(sub.to_string(index=False))


if __name__ == "__main__":
    main()
