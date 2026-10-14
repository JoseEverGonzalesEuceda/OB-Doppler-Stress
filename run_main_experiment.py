from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import ExperimentConfig
from ob_doppler import (
    augment_once,
    build_models,
    classification_metrics,
    load_dataset,
    set_seed,
    shift_linear,
    validate_manuscript_matrix,
    write_run_metadata,
)


def run(data_path: Path, output_dir: Path, cfg: ExperimentConfig, validate: bool = True):
    spectra, wavelength, y, star_names, subtypes, _ = load_dataset(data_path)
    if validate:
        dataset_info = validate_manuscript_matrix(spectra, wavelength, y)
    else:
        dataset_info = {
            "n_stars": int(len(spectra)),
            "n_B": int(np.sum(y == 0)),
            "n_O": int(np.sum(y == 1)),
            "n_wavelengths": int(spectra.shape[1]),
            "wavelength_min_A": float(wavelength[0]),
            "wavelength_max_A": float(wavelength[-1]),
            "median_step_A": float(np.median(np.diff(wavelength))),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    prediction_rows = []
    split_rows = []

    for repeat, seed in enumerate(cfg.seeds, start=1):
        set_seed(seed)
        indices = np.arange(len(spectra))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=cfg.test_size,
            stratify=y,
            random_state=seed,
        )

        X_train, X_test = spectra[train_idx], spectra[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        names_test = star_names[test_idx]
        subtypes_test = subtypes[test_idx]

        for i in train_idx:
            split_rows.append(
                {"repeat": repeat, "seed": seed, "star_name": star_names[i], "split": "train", "y_true": int(y[i])}
            )
        for i in test_idx:
            split_rows.append(
                {"repeat": repeat, "seed": seed, "star_name": star_names[i], "split": "test", "y_true": int(y[i])}
            )

        # Reproduces the augmentation draw used in the definitive run.
        rng = np.random.default_rng(seed * 1000 + 1)
        train_velocities = rng.uniform(
            -cfg.augmentation_vmax_kms,
            cfg.augmentation_vmax_kms,
            size=len(X_train),
        )
        X_aug, y_aug = augment_once(
            X_train,
            y_train,
            wavelength,
            train_velocities,
            operator=shift_linear,
            fill_value=cfg.fill_value,
        )

        training_sets = {
            "raw": (X_train, y_train),
            "raw_aug": (X_aug, y_aug),
        }

        for strategy, (X_fit, y_fit) in training_sets.items():
            models = build_models(seed, cfg.n_jobs, include_decision_tree=True)
            for model_name, model in models.items():
                print(f"seed={seed:3d} strategy={strategy:7s} model={model_name}")
                model.fit(X_fit, y_fit)

                for velocity in cfg.velocities_kms:
                    X_eval = (
                        X_test
                        if velocity == 0
                        else shift_linear(X_test, wavelength, velocity, cfg.fill_value)
                    )
                    y_pred = model.predict(X_eval)
                    p_O = model.predict_proba(X_eval)[:, 1]

                    row = {
                        "repeat": repeat,
                        "seed": seed,
                        "strategy": strategy,
                        "model": model_name,
                        "velocity_kms": float(velocity),
                        "abs_velocity_kms": abs(float(velocity)),
                    }
                    row.update(classification_metrics(y_test, y_pred, p_O))
                    metric_rows.append(row)

                    prediction_rows.extend(
                        {
                            "repeat": repeat,
                            "seed": seed,
                            "strategy": strategy,
                            "model": model_name,
                            "velocity_kms": float(velocity),
                            "abs_velocity_kms": abs(float(velocity)),
                            "star_name": str(names_test[j]),
                            "subtype": str(subtypes_test[j]),
                            "y_true": int(y_test[j]),
                            "y_pred": int(y_pred[j]),
                            "p_O": float(p_O[j]),
                        }
                        for j in range(len(y_test))
                    )

    pd.DataFrame(metric_rows).to_csv(output_dir / "metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(
        output_dir / "predictions.csv.gz", index=False, compression="gzip"
    )
    pd.DataFrame(split_rows).drop_duplicates().to_csv(
        output_dir / "split_manifest.csv", index=False
    )
    write_run_metadata(output_dir, cfg, dataset_info)


def main():
    parser = argparse.ArgumentParser(
        description="Run the Raw versus Raw+Aug Doppler-shift stress test."
    )
    parser.add_argument("--data", required=True, type=Path, help="Private dataset.npz file")
    parser.add_argument("--output", default=Path("results/main"), type=Path)
    parser.add_argument(
        "--skip-manuscript-validation",
        action="store_true",
        help="Allow a matrix that differs from the published 1455-star input.",
    )
    args = parser.parse_args()

    run(
        args.data,
        args.output,
        ExperimentConfig(),
        validate=not args.skip_manuscript_validation,
    )


if __name__ == "__main__":
    main()
